#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + ADX regime filter using 1d HTF
# Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# ADX > 25 indicates trending market (use 1d ADX for regime)
# In trending regime (ADX > 25): go long when Bull Power > 0 and rising, short when Bear Power < 0 and falling
# In ranging regime (ADX <= 25): fade extremes - long when Bear Power < -std and turning up, short when Bull Power > +std and turning down
# Uses 1d EMA(13) and ADX(14) for regime detection, 6h for entry timing
# Position size 0.25 to limit drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Works in both bull/bear: adapts to regime via ADX filter

name = "6h_1d_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data ONCE before loop for EMA and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA(13) for Elder Ray
    close_1d = df_1d['close'].values
    ema_13 = np.full(len(df_1d), np.nan)
    multiplier = 2 / (13 + 1)
    ema_13[0] = close_1d[0]
    for i in range(1, len(df_1d)):
        ema_13[i] = (close_1d[i] * multiplier) + (ema_13[i-1] * (1 - multiplier))
    
    # Calculate 1d ADX(14) for regime detection
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # True Range
    tr = np.zeros(len(df_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr0 = high_1d[i] - low_1d[i]
        tr1 = abs(high_1d[i] - close_1d[i-1])
        tr2 = abs(low_1d[i] - close_1d[i-1])
        tr[i] = max(tr0, tr1, tr2)
    
    # Directional Movement
    plus_dm = np.zeros(len(df_1d))
    minus_dm = np.zeros(len(df_1d))
    for i in range(1, len(df_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    # Smoothed DM and TR (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full(len(data), np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    # Calculate smoothed values
    tr_14 = wilders_smoothing(tr, 14)
    plus_dm_14 = wilders_smoothing(plus_dm, 14)
    minus_dm_14 = wilders_smoothing(minus_dm, 14)
    
    # Calculate DI and DX
    plus_di_14 = np.full(len(df_1d), np.nan)
    minus_di_14 = np.full(len(df_1d), np.nan)
    dx_14 = np.full(len(df_1d), np.nan)
    
    for i in range(14, len(df_1d)):
        if tr_14[i] != 0:
            plus_di_14[i] = (plus_dm_14[i] / tr_14[i]) * 100
            minus_di_14[i] = (minus_dm_14[i] / tr_14[i]) * 100
            if (plus_di_14[i] + minus_di_14[i]) != 0:
                dx_14[i] = (abs(plus_di_14[i] - minus_di_14[i]) / (plus_di_14[i] + minus_di_14[i])) * 100
    
    # Calculate ADX (smoothed DX)
    adx_14 = wilders_smoothing(dx_14, 14)
    
    # Align 1d data to 6h timeframe
    ema_13_6h = align_htf_to_ltf(prices, df_1d, ema_13)
    adx_14_6h = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate Elder Ray on 6h
    bull_power = high - ema_13_6h
    bear_power = low - ema_13_6h
    
    # Calculate standard deviation of bear/bull power for regime-based thresholds
    bull_power_ma = np.full(n, np.nan)
    bear_power_ma = np.full(n, np.nan)
    bull_power_std = np.full(n, np.nan)
    bear_power_std = np.full(n, np.nan)
    
    for i in range(20, n):
        bull_power_ma[i] = np.nanmean(bull_power[max(0, i-20):i+1])
        bear_power_ma[i] = np.nanmean(bear_power[max(0, i-20):i+1])
        bull_power_std[i] = np.nanstd(bull_power[max(0, i-20):i+1])
        bear_power_std[i] = np.nanstd(bear_power[max(0, i-20):i+1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_13_6h[i]) or 
            np.isnan(adx_14_6h[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or
            np.isnan(bull_power_std[i]) or
            np.isnan(bear_power_std[i])):
            signals[i] = 0.0
            continue
        
        adx = adx_14_6h[i]
        bp = bull_power[i]
        br = bear_power[i]
        bp_std = bull_power_std[i]
        br_std = bear_power_std[i]
        
        if position == 1:  # Long position
            # Exit conditions
            if adx > 25:  # Trending regime
                # Exit when bull power turns negative
                if bp <= 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # Ranging regime
                # Exit when price returns to mean (bull power crosses zero)
                if bp >= 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit conditions
            if adx > 25:  # Trending regime
                # Exit when bear power turns positive
                if br >= 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # Ranging regime
                # Exit when price returns to mean (bear power crosses zero)
                if br <= 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            # Entry logic based on regime
            if adx > 25:  # Trending regime - follow momentum
                # Go long when bull power is positive and rising
                # Go short when bear power is negative and falling
                if i > 30:
                    bp_prev = bull_power[i-1]
                    br_prev = bear_power[i-1]
                    if bp > 0 and bp > bp_prev:
                        position = 1
                        signals[i] = 0.25
                    elif br < 0 and br < br_prev:
                        position = -1
                        signals[i] = -0.25
            else:  # Ranging regime - mean reversion
                # Go long when bear power is extremely negative and turning up
                # Go short when bull power is extremely positive and turning down
                if i > 30:
                    br_prev = bear_power[i-1]
                    bp_prev = bull_power[i-1]
                    # Long: bear power below -1 std and turning up
                    if br < -br_std and br > br_prev:
                        position = 1
                        signals[i] = 0.25
                    # Short: bull power above +1 std and turning down
                    elif bp > bp_std and bp < bp_prev:
                        position = -1
                        signals[i] = -0.25
    
    return signals
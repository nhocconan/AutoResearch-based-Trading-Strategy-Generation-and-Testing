#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + volume confirmation + 1w ADX regime filter
# In trending regime (1w ADX > 25): trade breakouts in direction of trend (price > EMA50)
# In ranging regime (1w ADX <= 25): fade Donchian extremes (mean reversion at channel bounds)
# Uses 1w EMA(50) for trend filter and 1w ADX(14) for regime detection
# Position size 0.25 to limit drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Works in both bull/bear: adapts to regime via 1w ADX filter

name = "12h_1w_donchian_adx_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50 = np.full(len(df_1w), np.nan)
    multiplier = 2 / (50 + 1)
    ema_50[0] = close_1w[0]
    for i in range(1, len(df_1w)):
        ema_50[i] = (close_1w[i] * multiplier) + (ema_50[i-1] * (1 - multiplier))
    
    # Calculate 1w ADX(14) for regime detection
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # True Range
    tr = np.zeros(len(df_1w))
    tr[0] = high_1w[0] - low_1w[0]
    for i in range(1, len(df_1w)):
        tr0 = high_1w[i] - low_1w[i]
        tr1 = abs(high_1w[i] - close_1w[i-1])
        tr2 = abs(low_1w[i] - close_1w[i-1])
        tr[i] = max(tr0, tr1, tr2)
    
    # Directional Movement
    plus_dm = np.zeros(len(df_1w))
    minus_dm = np.zeros(len(df_1w))
    for i in range(1, len(df_1w)):
        up_move = high_1w[i] - high_1w[i-1]
        down_move = low_1w[i-1] - low_1w[i]
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
    plus_di_14 = np.full(len(df_1w), np.nan)
    minus_di_14 = np.full(len(df_1w), np.nan)
    dx_14 = np.full(len(df_1w), np.nan)
    
    for i in range(14, len(df_1w)):
        if tr_14[i] != 0:
            plus_di_14[i] = (plus_dm_14[i] / tr_14[i]) * 100
            minus_di_14[i] = (minus_dm_14[i] / tr_14[i]) * 100
            if (plus_di_14[i] + minus_di_14[i]) != 0:
                dx_14[i] = (abs(plus_di_14[i] - minus_di_14[i]) / (plus_di_14[i] + minus_di_14[i])) * 100
    
    # Calculate ADX (smoothed DX)
    adx_14 = wilders_smoothing(dx_14, 14)
    
    # Align 1w data to 12h timeframe
    ema_50_12h = align_htf_to_ltf(prices, df_1w, ema_50)
    adx_14_12h = align_htf_to_ltf(prices, df_1w, adx_14)
    
    # Calculate Donchian channels on 12h
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i+1])
        lowest_low[i] = np.min(low[i-20:i+1])
    
    # Calculate volume confirmation (volume > 1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i+1])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_50_12h[i]) or 
            np.isnan(adx_14_12h[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        adx = adx_14_12h[i]
        price = close[i]
        hh = highest_high[i]
        ll = lowest_low[i]
        vol_spike = volume_spike[i]
        trend_up = price > ema_50_12h[i]
        
        if position == 1:  # Long position
            # Exit conditions
            if adx > 25:  # Trending regime
                # Exit when price breaks below Donchian low OR trend reverses
                if price <= ll or not trend_up:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # Ranging regime
                # Exit when price returns to mean (mid-channel)
                mid = (hh + ll) / 2
                if price >= mid:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit conditions
            if adx > 25:  # Trending regime
                # Exit when price breaks above Donchian high OR trend reverses
                if price >= hh or trend_up:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # Ranging regime
                # Exit when price returns to mean (mid-channel)
                mid = (hh + ll) / 2
                if price <= mid:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            # Entry logic based on regime
            if adx > 25:  # Trending regime - follow momentum
                # Go long on breakout above Donchian high with volume and uptrend
                # Go short on breakdown below Donchian low with volume and downtrend
                if price >= hh and vol_spike and trend_up:
                    position = 1
                    signals[i] = 0.25
                elif price <= ll and vol_spike and not trend_up:
                    position = -1
                    signals[i] = -0.25
            else:  # Ranging regime - mean reversion
                # Go long at Donchian low with volume
                # Go short at Donchian high with volume
                if price <= ll and vol_spike:
                    position = 1
                    signals[i] = 0.25
                elif price >= hh and vol_spike:
                    position = -1
                    signals[i] = -0.25
    
    return signals
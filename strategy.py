#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d ADX regime filter
# Williams Alligator: Jaw (SMA13), Teeth (SMA8), Lips (SMA5) - all shifted forward by 5,3,2 bars
# ADX > 25 indicates trending market (use 1d ADX for regime)
# In trending regime (ADX > 25): trade Alligator alignment - long when Lips>Teeth>Jaw, short when Lips<Teeth<Jaw
# In ranging regime (ADX <= 25): fade extremes - long when price touches lower Bollinger(20,2), short when touches upper
# Uses 1d EMA(13) for trend bias and 1d ADX(14) for regime detection
# Position size 0.25 to limit drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Works in both bull/bear: adapts to regime via ADX filter

name = "12h_1d_williams_alligator_adx_regime_v1"
timeframe = "12h"
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
    
    # Calculate 1d EMA(13) for trend bias
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
    
    # Align 1d data to 12h timeframe
    ema_13_12h = align_htf_to_ltf(prices, df_1d, ema_13)
    adx_14_12h = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate Williams Alligator on 12h
    # Jaw: SMA(13) shifted by 8 bars
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: SMA(8) shifted by 5 bars
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: SMA(5) shifted by 3 bars
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate Bollinger Bands for ranging regime
    bb_period = 20
    bb_std = 2
    bb_ma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_ma + (bb_std_dev * bb_std)
    bb_lower = bb_ma - (bb_std_dev * bb_std)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_13_12h[i]) or 
            np.isnan(adx_14_12h[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or
            np.isnan(lips[i]) or
            np.isnan(bb_ma[i]) or
            np.isnan(bb_upper[i]) or
            np.isnan(bb_lower[i])):
            signals[i] = 0.0
            continue
        
        adx = adx_14_12h[i]
        ema_bias = ema_13_12h[i]
        lip = lips[i]
        tee = teeth[i]
        jaw_val = jaw[i]
        bb_up = bb_upper[i]
        bb_low = bb_lower[i]
        price = close[i]
        
        if position == 1:  # Long position
            # Exit conditions
            if adx > 25:  # Trending regime
                # Exit when Alligator alignment breaks (Lips < Teeth)
                if lip <= tee:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # Ranging regime
                # Exit when price returns to mean (crosses above BB middle)
                if price >= bb_ma:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit conditions
            if adx > 25:  # Trending regime
                # Exit when Alligator alignment breaks (Lips > Teeth)
                if lip >= tee:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # Ranging regime
                # Exit when price returns to mean (crosses below BB middle)
                if price <= bb_ma:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            # Entry logic based on regime
            if adx > 25:  # Trending regime - follow Alligator alignment
                # Go long when Alligator is aligned bullish (Lips > Teeth > Jaw)
                # Go short when Alligator is aligned bearish (Lips < Teeth < Jaw)
                if lip > tee and tee > jaw_val:
                    position = 1
                    signals[i] = 0.25
                elif lip < tee and tee < jaw_val:
                    position = -1
                    signals[i] = -0.25
            else:  # Ranging regime - mean reversion at Bollinger Bands
                # Go long when price touches lower BB and EMA bias is up
                # Go short when price touches upper BB and EMA bias is down
                if price <= bb_low and close[i] > ema_bias:
                    position = 1
                    signals[i] = 0.25
                elif price >= bb_up and close[i] < ema_bias:
                    position = -1
                    signals[i] = -0.25
    
    return signals
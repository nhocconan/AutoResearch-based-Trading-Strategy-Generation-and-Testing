#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band breakout with 1w EMA50 trend filter and volume confirmation
# In strong trends (price > weekly EMA50), we take breakouts from Bollinger Bands (20,2) for trend continuation.
# Volume spike (>1.5x 20 EMA) confirms institutional participation. Discrete sizing 0.25 limits risk.
# Target: 30-100 total trades over 4 years (7-25/year). Works in bull/bear: uses trend filter to align with higher timeframe direction.

name = "1d_BollingerBreakout_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend direction
    close_1w = pd.Series(df_1w['close'])
    ema50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe (completed 1w bar only)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Bollinger Bands on 1d timeframe
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean()
    dev = 2.0 * close_s.rolling(window=20, min_periods=20).std()
    upper_band = basis + dev
    lower_band = basis - dev
    upper_band = upper_band.values
    lower_band = lower_band.values
    
    # Volume confirmation: 20-period EMA of volume on 1d timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_aligned[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Long conditions: price breaks above upper band + uptrend + volume spike
            if close[i] > upper_band[i] and close[i] > ema50_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower band + downtrend + volume spike
            elif close[i] < lower_band[i] and close[i] < ema50_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to basis OR trend changes OR volume drops
            if (close[i] < basis.iloc[i] or 
                close[i] < ema50_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to basis OR trend changes OR volume drops
            if (close[i] > basis.iloc[i] or 
                close[i] > ema50_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
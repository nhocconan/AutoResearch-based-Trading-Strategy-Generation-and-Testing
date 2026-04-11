#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_pivot_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Calculate Camarilla pivot levels (H3, L3) from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot and range from previous day
    pivot = (high_1d + low_1d + close_1d) / 3
    range_ = high_1d - low_1d
    
    # Camarilla levels: H3/L3 for day trading
    h3 = close_1d + range_ * 1.1 / 4
    l3 = close_1d - range_ * 1.1 / 4
    
    # Shift by 1 to use only completed daily bars
    h3 = np.roll(h3, 1)
    l3 = np.roll(l3, 1)
    h3[0] = np.nan
    l3[0] = np.nan
    
    # Align daily levels to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume filter: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop loss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - low[:-1])
    tr3 = np.abs(low[1:] - high[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.8 * vol_ma
        
        # Long when price touches or breaks above H3 with volume
        long_signal = volume_confirmed and price_high >= h3_aligned[i]
        
        # Short when price touches or breaks below L3 with volume
        short_signal = volume_confirmed and price_low <= l3_aligned[i]
        
        # Exit conditions
        exit_long = position == 1 and price_close < h3_aligned[i] - 0.5 * atr_val
        exit_short = position == -1 and price_close > l3_aligned[i] + 0.5 * atr_val
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Camarilla H3/L3 levels act as intraday support/resistance. 
# Price touching H3 with volume indicates bullish breakout; touching L3 indicates bearish breakout.
# Uses 1d Camarilla levels (H3/L3) for intraday reference, volume confirmation to filter weak moves,
# and ATR-based exit to manage risk. Works in both bull and bear markets by capturing
# breakouts from key levels. Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
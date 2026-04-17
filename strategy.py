#!/usr/bin/env python3
"""
4h Camarilla Pivot + 1d EMA Trend + Volume Spike
Long: Price > Camarilla H3 + price > 1d EMA(34) + volume > 1.5x 4h volume SMA(20)
Short: Price < Camarilla L3 + price < 1d EMA(34) + volume > 1.5x 4h volume SMA(20)
Exit: Price crosses back through Camarilla H3/L3 or EMA(34) flip
Uses Camarilla levels for institutional support/resistance, EMA for trend filter, volume for confirmation.
Designed to capture breakouts with institutional levels in both bull and bear markets.
Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(34)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h volume SMA(20) for volume filter
    vol_sma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(34, 20)  # need EMA and volume SMA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_sma_4h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        vol_sma_val = vol_sma_4h[i]
        ema_val = ema_34_1d_aligned[i]
        
        # Calculate Camarilla levels from previous day's range
        # Need previous day's high, low, close
        if i >= 1:
            # Get indices for previous day's data (assuming 24h periods)
            # Since we're on 4h timeframe, previous day = 6 bars ago
            prev_day_idx = max(0, i - 6)
            if prev_day_idx < len(high):
                prev_high = high[prev_day_idx]
                prev_low = low[prev_day_idx]
                prev_close = close[prev_day_idx]
            else:
                # Use available data
                prev_high = high[0]
                prev_low = low[0]
                prev_close = close[0]
            
            # Camarilla levels
            range_val = prev_high - prev_low
            if range_val > 0:
                H3 = prev_close + range_val * 1.1 / 4
                L3 = prev_close - range_val * 1.1 / 4
                H4 = prev_close + range_val * 1.1 / 2
                L4 = prev_close - range_val * 1.1 / 2
            else:
                H3 = L3 = H4 = L4 = prev_close
        else:
            H3 = L3 = H4 = L4 = close[0]
        
        if position == 0:
            # Long: Price > H3 + price > EMA(34) + volume spike
            if price > H3 and price > ema_val and vol > 1.5 * vol_sma_val:
                signals[i] = 0.25
                position = 1
            # Short: Price < L3 + price < EMA(34) + volume spike
            elif price < L3 and price < ema_val and vol > 1.5 * vol_sma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price < H3 or price < EMA(34)
            if price < H3 or price < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price > L3 or price > EMA(34)
            if price > L3 or price > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0
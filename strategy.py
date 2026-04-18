#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian_20_Volume_Trend_HTF12h"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Trend filter: 12h EMA34
    df_12h = get_htf_data(prices, '12h')
    ema34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # ATR for stop loss
    tr1 = high - np.roll(low, 1)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(ema34_12h_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        hh_val = highest_high[i]
        ll_val = lowest_low[i]
        ema_val = ema34_12h_aligned[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: Break above Donchian high with volume and above 12h EMA34
            if high_val > hh_val and volume_filter[i] and close_val > ema_val:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: Break below Donchian low with volume and below 12h EMA34
            elif low_val < ll_val and volume_filter[i] and close_val < ema_val:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        
        elif position == 1:
            # Long exit: Stop loss or reverse signal
            if low_val <= entry_price - 2.0 * atr_val or low_val < ll_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Stop loss or reverse signal
            if high_val >= entry_price + 2.0 * atr_val or high_val > hh_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
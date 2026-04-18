#!/usr/bin/env python3
"""
1h_BollingerBreakout_TrailingStop
Hypothesis: Buy when price breaks above upper Bollinger Band with volume > 1.5x average and close > 1h EMA20; short when breaks below lower band with volume > 1.5x average and close < 1h EMA20. Use 1h EMA20 as dynamic trailing stop. Bollinger Bands capture volatility expansion, volume confirms institutional interest, EMA20 filters counter-trend moves. Designed for low trade frequency with clear exit rules.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2.0)
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean().values
    dev = close_s.rolling(window=20, min_periods=20).std().values
    upper = basis + 2.0 * dev
    lower = basis - 2.0 * dev
    
    # Volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # 1h EMA20 for trend filter and trailing stop
    ema20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = 20  # Need Bollinger and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(upper[i]) or 
            np.isnan(lower[i]) or
            np.isnan(volume_filter[i]) or
            np.isnan(ema20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ok = volume_filter[i]
        ema_val = ema20[i]
        upper_val = upper[i]
        lower_val = lower[i]
        
        if position == 0:
            # Long: break above upper band with volume filter and above EMA20
            if price > upper_val and vol_ok and price > ema_val:
                signals[i] = 0.20
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: break below lower band with volume filter and below EMA20
            elif price < lower_val and vol_ok and price < ema_val:
                signals[i] = -0.20
                position = -1
                entry_price = price
                lowest_since_entry = price
        
        elif position == 1:
            signals[i] = 0.20
            highest_since_entry = max(highest_since_entry, price)
            # Exit: trailing stop (2.0 * ATR-like using price range) OR close below EMA20
            # Use 2.0 * (highest_since_entry - lowest_since_entry) as dynamic stop distance
            if len(range(max(0, i-5), i+1)) >= 2:
                recent_high = np.max(high[max(0, i-5):i+1])
                recent_low = np.min(low[max(0, i-5):i+1])
                range_val = recent_high - recent_low
                trail_stop = highest_since_entry - 1.5 * range_val
                if price < trail_stop or price < ema_val:
                    signals[i] = 0.0
                    position = 0
        
        elif position == -1:
            signals[i] = -0.20
            lowest_since_entry = min(lowest_since_entry, price)
            # Exit: trailing stop OR close above EMA20
            if len(range(max(0, i-5), i+1)) >= 2:
                recent_high = np.max(high[max(0, i-5):i+1])
                recent_low = np.min(low[max(0, i-5):i+1])
                range_val = recent_high - recent_low
                trail_stop = lowest_since_entry + 1.5 * range_val
                if price > trail_stop or price > ema_val:
                    signals[i] = 0.0
                    position = 0
    
    return signals

name = "1h_BollingerBreakout_TrailingStop"
timeframe = "1h"
leverage = 1.0
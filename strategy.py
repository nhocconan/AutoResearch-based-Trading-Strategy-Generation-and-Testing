#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian channel breakout with 4h volume confirmation and 1d EMA50 trend filter.
# Long when: Price breaks above 20-period Donchian high, volume > 2x 4h average, price > 1d EMA50
# Short when: Price breaks below 20-period Donchian low, volume > 2x 4h average, price < 1d EMA50
# Exit when: Price returns to Donchian mean (midpoint)
# Donchian provides clear breakout levels, volume confirms conviction, EMA50 filters trend.
# Target: 15-30 trades/year per symbol. Works in bull (buy breakouts) and bear (sell breakdowns).
name = "1h_Donchian20_4hVol_1dEMA50"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1h Donchian channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_4h_aligned[i]
        ema50 = ema50_1d_aligned[i]
        upper = highest_high[i]
        lower = lowest_low[i]
        mid = donchian_mid[i]
        
        if position == 0:
            # Long entry: break above Donchian high, volume spike, above EMA50
            if (price > upper and vol > 2.0 * vol_ma and price > ema50):
                signals[i] = 0.20
                position = 1
            # Short entry: break below Donchian low, volume spike, below EMA50
            elif (price < lower and vol > 2.0 * vol_ma and price < ema50):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: return to Donchian midpoint
            if price < mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: return to Donchian midpoint
            if price > mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
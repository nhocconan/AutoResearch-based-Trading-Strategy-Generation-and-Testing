#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d trend filter + volume confirmation
# Uses 6h Donchian channels for breakouts, with 1d EMA200 trend filter and volume spike.
# Long when price breaks above Donchian upper band with volume > 2x average and 1d close > EMA200.
# Short when price breaks below Donchian lower band with volume > 2x average and 1d close < EMA200.
# Exit when price returns to Donchian middle (mean) or volume drops below average.
# Designed for ~20-30 trades/year with strong filters to avoid whipsaws in choppy markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 6h Donchian channels (20-period)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    for i in range(19, n):
        upper[i] = np.max(high[i-19:i+1])
        lower[i] = np.min(low[i-19:i+1])
        middle[i] = (upper[i] + lower[i]) / 2.0
    
    # Volume filter: volume > 2x 24-period average (24*6h = 6 days)
    vol_ma_24 = np.full(n, np.nan)
    for i in range(23, n):
        vol_ma_24[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 20-period Donchian and 24-period volume MA
    start_idx = max(24, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or 
            np.isnan(ema200_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_24[i]
        
        # Volume filter
        vol_filter = vol_now > 2.0 * vol_avg
        
        # Trend filter from 1d EMA200
        bullish_trend = price > ema200_aligned[i]
        bearish_trend = price < ema200_aligned[i]
        
        if position == 0:
            # Long: break above upper band with volume and bullish trend
            if price > upper[i] and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: break below lower band with volume and bearish trend
            elif price < lower[i] and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: return to middle or volume drops
            if price < middle[i] or vol_now <= vol_avg:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: return to middle or volume drops
            if price > middle[i] or vol_now <= vol_avg:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_Breakout_1dEMA200_Volume"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# Long when price breaks above Donchian upper(20) and weekly pivot shows bullish bias
# Short when price breaks below Donchian lower(20) and weekly pivot shows bearish bias
# Weekly pivot: price above weekly VWAP = bullish, below = bearish
# Volume confirmation: current volume > 2.0 * average volume of last 20 periods
# Position size: 0.25 (25% of capital)
# Target: 60-150 total trades over 4 years (15-38/year)
# Uses weekly trend filter to avoid counter-trend trades in choppy markets

name = "6h_donchian20_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot/VWAP filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly VWAP for trend filter
    typical_price_1w = (df_1w['high'].values + df_1w['low'].values + df_1w['close'].values) / 3
    vwap_numerator = np.cumsum(typical_price_1w * df_1w['volume'].values)
    vwap_denominator = np.cumsum(df_1w['volume'].values)
    vwap_1w = vwap_numerator / vwap_denominator
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Average volume for volume confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(vwap_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below Donchian lower(20)
            elif close[i] < low[i-20]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above Donchian upper(20)
            elif close[i] > high[i-20]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Calculate Donchian channels (20-period)
            highest_high = high[i-20:i].max() if i >= 20 else high[:i].max()
            lowest_low = low[i-20:i].min() if i >= 20 else low[:i].min()
            
            # Weekly trend filter: price above/below weekly VWAP
            bullish = close[i] > vwap_1w_aligned[i]
            bearish = close[i] < vwap_1w_aligned[i]
            
            # Volume confirmation: current volume > 2.0 * average volume
            volume_confirm = volume[i] > 2.0 * vol_avg[i]
            
            # Long: price breaks above Donchian upper(20) in bullish weekly trend with volume
            if close[i] > highest_high and bullish and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian lower(20) in bearish weekly trend with volume
            elif close[i] < lowest_low and bearish and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals
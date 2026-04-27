#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + Donchian(20) breakout + volume confirmation
# Choppiness Index > 61.8 indicates ranging market (mean revert at Donchian bands)
# Choppiness Index < 38.2 indicates trending market (breakout in direction of trend)
# Works in bull/bear by adapting to market regime: mean revert in range, follow trend when trending.
# Uses volume confirmation to avoid false breakouts. Target: 20-50 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for 1d
    tr_1d = np.zeros(len(df_1d))
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr_1d[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    # Calculate Choppiness Index (14-period)
    atr_14_1d = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        atr_14_1d[i] = np.mean(tr_1d[i-13:i+1])
    
    # Sum of true ranges over 14 periods
    sum_tr_14 = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        sum_tr_14[i] = np.sum(tr_1d[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    max_high_14 = np.full(len(df_1d), np.nan)
    min_low_14 = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        max_high_14[i] = np.max(high_1d[i-13:i+1])
        min_low_14[i] = np.min(low_1d[i-13:i+1])
    
    # Choppiness Index formula: 100 * log10(sum(tr14) / (max_high - min_low)) / log10(14)
    chop = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        if max_high_14[i] > min_low_14[i] and sum_tr_14[i] > 0:
            chop[i] = 100 * np.log10(sum_tr_14[i] / (max_high_14[i] - min_low_14[i])) / np.log10(14)
        else:
            chop[i] = 50.0  # neutral when undefined
    
    # Align Choppiness Index to 4h timeframe (wait for 1d close)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian channels (20-period) on 4h data
    highest_high_20 = np.full(n, np.nan)
    lowest_low_20 = np.full(n, np.nan)
    for i in range(19, n):
        highest_high_20[i] = np.max(high[i-19:i+1])
        lowest_low_20[i] = np.min(low[i-19:i+1])
    
    # Volume filter: volume > 1.5 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need chop (14), Donchian (20), volume MA (20)
    start_idx = max(14, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(chop_aligned[i]) or 
            np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        chop_val = chop_aligned[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: significant volume
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Regime filters
        ranging_market = chop_val > 61.8  # choppy/ranging market
        trending_market = chop_val < 38.2  # trending market
        
        if position == 0:
            if ranging_market and vol_filter:
                # Mean reversion in ranging market: buy at lower band, sell at upper band
                if price <= lowest_low_20[i]:
                    signals[i] = size
                    position = 1
                elif price >= highest_high_20[i]:
                    signals[i] = -size
                    position = -1
                else:
                    signals[i] = 0.0
            elif trending_market and vol_filter:
                # Trend following in trending market: breakout in direction of trend
                if price > highest_high_20[i]:
                    signals[i] = size
                    position = 1
                elif price < lowest_low_20[i]:
                    signals[i] = -size
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: opposite Donchian band or regime change to ranging with reversal
            if price >= highest_high_20[i] or (ranging_market and price >= lowest_low_20[i] + (highest_high_20[i] - lowest_low_20[i]) * 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: opposite Donchian band or regime change to ranging with reversal
            if price <= lowest_low_20[i] or (ranging_market and price <= highest_high_20[i] - (highest_high_20[i] - lowest_low_20[i]) * 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Choppiness_Donchian_Breakout_Volume"
timeframe = "4h"
leverage = 1.0
#!/usr/bin/env python3
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
    
    # Get daily data for Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-day ATR
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], 
                       np.maximum(np.abs(high_1d[1:] - close_1d[:-1]),
                                  np.abs(low_1d[1:] - close_1d[:-1])))
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], tr_1d])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate sum of true ranges for numerator
    sum_tr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Calculate max(high) - min(low) over 14 days for denominator
    max_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14_1d = max_high_1d - min_low_1d
    
    # Choppiness Index: CHOP = 100 * log10(sum_tr / range) / log10(14)
    chop_1d = 100 * np.log10(sum_tr_1d / range_14_1d) / np.log10(14)
    
    # Chop > 61.8 = ranging market (mean revert), Chop < 38.2 = trending
    chop_range = chop_1d > 61.8
    chop_trend = chop_1d < 38.2
    
    # Align Chop indicators to 4h timeframe
    chop_range_aligned = align_htf_to_ltf(prices, df_1d, chop_range)
    chop_trend_aligned = align_htf_to_ltf(prices, df_1d, chop_trend)
    
    # Calculate 4h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 4h Bollinger Bands (20, 2)
    bb_middle = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Calculate volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(chop_range_aligned[i]) or 
            np.isnan(chop_trend_aligned[i]) or
            np.isnan(rsi[i]) or
            np.isnan(bb_upper[i]) or
            np.isnan(bb_lower[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Mean reversion in ranging markets: buy at lower BB, sell at upper BB
        long_entry = chop_range_aligned[i] and close[i] <= bb_lower[i] and volume[i] > vol_ma[i]
        short_entry = chop_range_aligned[i] and close[i] >= bb_upper[i] and volume[i] > vol_ma[i]
        
        # Exit: return to middle band
        long_exit = chop_range_aligned[i] and close[i] >= bb_middle[i]
        short_exit = chop_range_aligned[i] and close[i] <= bb_middle[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Chop_RSI_BB_MeanRev_Volume"
timeframe = "4h"
leverage = 1.0
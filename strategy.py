#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Choppiness Index regime filter + 1w EMA trend filter + volume breakout
# Long when: CHOP(14) > 61.8 (range) AND price > EMA50(1w) AND volume > 2x 20-day average
# Short when: CHOP(14) > 61.8 (range) AND price < EMA50(1w) AND volume > 2x 20-day average
# Exit when: CHOP(14) < 38.2 (trend) OR price crosses EMA50(1w) in opposite direction
# Uses chop to identify mean-reversion regimes, EMA for trend filter, volume for confirmation
# Target: 20-40 total trades over 4 years (5-10/year) to minimize fee drag

name = "1d_Chop_Range_EMA50_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 2x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma20)
    
    # 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # EMA50 on 1w close
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Choppiness Index (14) on daily data
    # CHOP = 100 * log10(sum(ATR(1)) / (max(high) - min(low))) / log10(14)
    tr1 = np.maximum(high - low, 
                     np.maximum(np.abs(high - np.roll(close, 1)), 
                                np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]  # first bar
    
    atr_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_maxmin = max_high - min_low
    
    # Avoid division by zero
    chop_raw = np.where(range_maxmin > 0, 
                        np.log10(atr_sum / range_maxmin) / np.log10(14) * 100, 
                        50.0)  # neutral when range is zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA and CHOP
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(chop_raw[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Entry conditions: range regime (CHOP > 61.8) + volume + EMA filter
            in_range = chop_raw[i] > 61.8
            
            if in_range and volume_filter[i]:
                # Long: price above weekly EMA50
                if close[i] > ema_50_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price below weekly EMA50
                elif close[i] < ema_50_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: trend regime (CHOP < 38.2) OR price crosses below EMA50
            if chop_raw[i] < 38.2 or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend regime (CHOP < 38.2) OR price crosses above EMA50
            if chop_raw[i] < 38.2 or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
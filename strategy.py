#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot levels with volume confirmation and chop regime filter
# Fade at R3/S3 (mean reversion) in choppy markets (CHOP > 61.8), breakout at R4/S4 in trending markets (CHOP < 38.2)
# Volume confirmation (current 4h volume > 1.5x 20-period average) filters false signals
# Position size fixed at 0.25 to balance return and drawdown
# Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)

name = "4h_1d_camarilla_chop_volume_v1"
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
    open_time = prices['open_time'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r4_1d = close_1d + range_1d * 1.1 / 2.0
    r3_1d = close_1d + range_1d * 1.1 / 4.0
    s3_1d = close_1d - range_1d * 1.1 / 4.0
    s4_1d = close_1d - range_1d * 1.1 / 2.0
    
    # Calculate 1d ATR (14-period) for choppiness index
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1_1d[0]  # First period has no previous close
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Choppiness Index (14-period) on 1d timeframe
    # CHOP = 100 * log10(sum(ATR14) / (max(high14) - min(low14))) / log10(14)
    sum_atr_14 = pd.Series(atr_14_1d).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    chop_raw = np.where(range_14 > 0, sum_atr_14 / range_14, 1.0)
    chop_1d = 100 * np.log10(chop_raw) / np.log10(14)
    
    # Align all HTF data to 4h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Pre-compute volume confirmation (20-period average for 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(chop_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 4h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if not volume_confirmed:
            signals[i] = 0.0
            continue
        
        # Fixed position size to minimize fee churn
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit on retracement to S3 or stop at S4 breakdown
            if close[i] < s3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close[i] < s4_1d_aligned[i]:  # Stop loss at S4 breakdown
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit on retracement to R3 or stop at R4 breakout
            if close[i] > r3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close[i] > r4_1d_aligned[i]:  # Stop loss at R4 breakout
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Camarilla pivot trading with volume confirmation and chop regime filter
            if chop_1d_aligned[i] > 61.8:  # Choppy market - mean reversion
                # Fade at R3 (sell at resistance, expect reversion to pivot)
                if close[i] > r3_1d_aligned[i] and close[i] < r4_1d_aligned[i]:
                    position = -1
                    signals[i] = -position_size
                # Fade at S3 (buy at support, expect reversion to pivot)
                elif close[i] < s3_1d_aligned[i] and close[i] > s4_1d_aligned[i]:
                    position = 1
                    signals[i] = position_size
            elif chop_1d_aligned[i] < 38.2:  # Trending market - breakout continuation
                # Breakout continuation at R4 (buy break above resistance)
                if close[i] > r4_1d_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Breakout continuation at S4 (sell break below support)
                elif close[i] < s4_1d_aligned[i]:
                    position = -1
                    signals[i] = -position_size
    
    return signals
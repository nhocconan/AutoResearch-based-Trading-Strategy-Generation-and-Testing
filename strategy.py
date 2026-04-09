#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with volume confirmation and chop regime filter
# Uses 4h Camarilla levels (R3/S3, R4/S4) from prior completed bar
# Fade at R3/S3 (mean reversion), breakout at R4/S4 (trend continuation)
# Volume confirmation requires current 4h volume > 1.5x 20-period average
# Chop regime filter uses 4h Choppiness Index (CHOP > 61.8 = range, CHOP < 38.2 = trend)
# In range: fade at R3/S3; in trend: breakout at R4/S4
# Position size: 0.25 to balance profit potential and fee drag
# Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)

name = "4h_1d_camarilla_vol_chop_v1"
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
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (from prior completed 1d bar)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r4_1d = close_1d + range_1d * 1.1 / 2.0
    r3_1d = close_1d + range_1d * 1.1 / 4.0
    s3_1d = close_1d - range_1d * 1.1 / 4.0
    s4_1d = close_1d - range_1d * 1.1 / 2.0
    
    # Calculate 4h ATR (14-period) for volatility filtering
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4h Choppiness Index (CHOP)
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high,n) - min(low,n))) / log10(n)
    # where n = 14 periods
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum()
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    chop = chop.values  # convert to numpy array
    
    # Align all HTF data to 4h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Pre-compute volume confirmation (20-period average for 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(atr_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(vol_ma_20[i]) or atr_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 4h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if not volume_confirmed:
            signals[i] = 0.0
            continue
        
        # Regime filter based on Choppiness Index
        chop_val = chop_aligned[i]
        in_range = chop_val > 61.8  # ranging market
        in_trend = chop_val < 38.2  # trending market
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit conditions: retracement to S3 or stop at S4 breakdown
            if close[i] < s3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close[i] < s4_1d_aligned[i]:  # Stop loss at S4 breakdown
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit conditions: retracement to R3 or stop at R4 breakout
            if close[i] > r3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close[i] > r4_1d_aligned[i]:  # Stop loss at R4 breakout
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Entry logic based on market regime
            if in_range:
                # In ranging market: fade at R3/S3 (mean reversion)
                if close[i] > r3_1d_aligned[i] and close[i] < r4_1d_aligned[i]:
                    # Sell at R3 resistance, expect reversion to pivot
                    position = -1
                    signals[i] = -position_size
                elif close[i] < s3_1d_aligned[i] and close[i] > s4_1d_aligned[i]:
                    # Buy at S3 support, expect reversion to pivot
                    position = 1
                    signals[i] = position_size
            elif in_trend:
                # In trending market: breakout at R4/S4 (trend continuation)
                if close[i] > r4_1d_aligned[i]:
                    # Buy break above R4 resistance
                    position = 1
                    signals[i] = position_size
                elif close[i] < s4_1d_aligned[i]:
                    # Sell break below S4 support
                    position = -1
                    signals[i] = -position_size
    
    return signals
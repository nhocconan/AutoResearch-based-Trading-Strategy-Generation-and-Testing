#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot R3/S3 breakout with 1d volume spike and ADX regime filter
# Uses Camarilla pivot levels from 1d to identify key support/resistance levels.
# Enters on breakout of R3 (long) or S3 (short) with 1d volume confirmation (>1.5x 20-period average).
# Uses 1d ADX > 25 to filter for trending markets and avoid whipsaw in ranging conditions.
# Designed for 20-50 trades/year (~80-200 total over 4 years) to minimize fee drag.
# Camarilla pivots work well in both bull/bear markets by adapting to volatility.
# Volume spike ensures institutional participation. ADX filter avoids false breakouts.

name = "4h_Camarilla_R3S3_Breakout_1dVolumeSpike_ADXRegime"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots, volume, and ADX - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA20 for volume average
    ema20_vol_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1d True Range for ADX
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d ADX components
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 4h timeframe (wait for completed 1d bar)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    volume_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_vol_1d)
    
    # Calculate Camarilla pivot levels for 1d (based on previous day's OHLC)
    # Camarilla: R4 = close + ((high-low)*1.1/2), R3 = close + ((high-low)*1.1/4),
    #            S3 = close - ((high-low)*1.1/4), S4 = close - ((high-low)*1.1/2)
    # We use R3/S3 as breakout levels
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = np.nan
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    rang = prev_high_1d - prev_low_1d
    r3_1d = prev_close_1d + (rang * 1.1 / 4)
    s3_1d = prev_close_1d - (rang * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(volume_avg_1d_aligned[i]) or
            np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current 1d volume > 1.5x 20-period average
        # We approximate current 1d volume as the volume of the completed 1d bar
        volume_condition = volume_1d[i//16] > 1.5 * volume_avg_1d_aligned[i] if i//16 < len(volume_1d) else False
        
        # ADX condition: trending market (ADX > 25)
        adx_condition = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Long conditions: price breaks above R3 with volume and ADX confirmation
            if (close[i] > r3_1d_aligned[i] and volume_condition and adx_condition):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3 with volume and ADX confirmation
            elif (close[i] < s3_1d_aligned[i] and volume_condition and adx_condition):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters below R3 OR ADX weakens (< 20) OR volume drops
            if (close[i] < r3_1d_aligned[i] or adx_1d_aligned[i] < 20 or 
                (i//16 < len(volume_1d) and volume_1d[i//16] < volume_avg_1d_aligned[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters above S3 OR ADX weakens OR volume drops
            if (close[i] > s3_1d_aligned[i] or adx_1d_aligned[i] < 20 or
                (i//16 < len(volume_1d) and volume_1d[i//16] < volume_avg_1d_aligned[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
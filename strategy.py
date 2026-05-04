#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and chop regime filter
# In choppy markets (CHOP>61.8), we fade extremes: short near R3, long near S3 with volume confirmation
# In trending markets (CHOP<38.2), we trade breakouts: long on R3 breakout, short on S3 breakout
# Uses 1d timeframe for CHOP regime and volume spike confirmation to reduce false signals
# Designed for 12h timeframe targeting 50-150 total trades over 4 years with discrete sizing (0.25)

name = "12h_Camarilla_R3S3_1dChop_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for CHOP regime and volume spike
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d CHOP (Choppiness Index) - 14 period
    tr1 = pd.Series(df_1d['high']).sub(df_1d['low'])
    tr2 = pd.Series(df_1d['high']).sub(df_1d['close'].shift(1)).abs()
    tr3 = pd.Series(df_1d['low']).sub(df_1d['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = tr.rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop_values = chop.values
    
    # Calculate 1d volume EMA (20-period) for spike detection
    vol_ema_20_1d = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 12h Camarilla levels (R3, S3) based on previous 12h bar
    # Camarilla: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    # Using rolling window of 1 for previous bar values
    prev_close = pd.Series(close).shift(1)
    prev_high = pd.Series(high).shift(1)
    prev_low = pd.Series(low).shift(1)
    camarilla_range = prev_high - prev_low
    r3 = prev_close + 1.1 * camarilla_range * 1.1 / 4
    s3 = prev_close - 1.1 * camarilla_range * 1.1 / 4
    
    # Align 1d indicators to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    vol_ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_20_1d)
    
    # 12h volume EMA (20-period) for confirmation
    vol_ema_20_12h = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(vol_ema_20_1d_aligned[i]) or 
            np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(vol_ema_20_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: 12h volume > 1.5 x 20-period EMA AND 1d volume > 1.5 x 20-period EMA
        vol_confirm_12h = volume[i] > (1.5 * vol_ema_20_12h[i])
        vol_confirm_1d = df_1d['volume'].iloc[-1] > (1.5 * vol_ema_20_1d_aligned[i]) if len(df_1d) > 0 else False
        volume_confirm = vol_confirm_12h and vol_confirm_1d
        
        if position == 0:
            # Determine regime: choppy (CHOP>61.8) or trending (CHOP<38.2)
            if chop_aligned[i] > 61.8:
                # Choppy market: fade extremes (mean reversion)
                if close[i] <= s3[i] and volume_confirm:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= r3[i] and volume_confirm:
                    signals[i] = -0.25
                    position = -1
            elif chop_aligned[i] < 38.2:
                # Trending market: trade breakouts
                if close[i] > r3[i] and volume_confirm:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < s3[i] and volume_confirm:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price retouches midpoint OR chop increases (>50) OR volume drops
            midpoint = (r3[i] + s3[i]) / 2
            if (close[i] >= midpoint or 
                chop_aligned[i] > 50 or 
                volume[i] < vol_ema_20_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches midpoint OR chop increases (>50) OR volume drops
            midpoint = (r3[i] + s3[i]) / 2
            if (close[i] <= midpoint or 
                chop_aligned[i] > 50 or 
                volume[i] < vol_ema_20_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
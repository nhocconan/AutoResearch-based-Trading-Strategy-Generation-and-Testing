#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and 1d chop regime filter
# - Uses daily Camarilla levels (R3, R4, S3, S4) from previous day
# - Long when price breaks above R4 with volume > 1.8x 20-period average AND chop > 61.8 (range)
# - Short when price breaks below S4 with volume > 1.8x 20-period average AND chop > 61.8 (range)
# - Exit when price retests to R3/S3 OR chop < 38.2 (trend) OR volume < average
# - Chop filter ensures we only trade breakouts in ranging markets where false breakouts reverse
# - Targets 12-25 trades/year (50-100 total over 4 years) to stay within fee drag limits
# - Works in both bull/bear: range markets produce mean-reverting breakouts; chop filter adapts

name = "12h_1d_camarilla_breakout_chop_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute daily Camarilla pivots (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day (using previous day's data)
    camarilla_r4 = np.full_like(close_1d, np.nan)
    camarilla_r3 = np.full_like(close_1d, np.nan)
    camarilla_s3 = np.full_like(close_1d, np.nan)
    camarilla_s4 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        # Previous day's range
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_val = prev_high - prev_low
        
        if range_val > 0:
            camarilla_r4[i] = prev_close + range_val * 1.1 / 2
            camarilla_r3[i] = prev_close + range_val * 1.1 / 4
            camarilla_s3[i] = prev_close - range_val * 1.1 / 4
            camarilla_s4[i] = prev_close - range_val * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Pre-compute volume confirmation: > 1.8x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.8 * volume_20_avg)
    
    # Pre-compute volume normal: < average volume for exit
    vol_normal = prices['volume'] < volume_20_avg
    
    # Pre-compute 1d Chopiness Index (14-period) for regime filter
    high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    close_14 = pd.Series(close_1d).rolling(window=14, min_periods=14).sum().values
    atr_14 = pd.Series(high_1d - low_1d).rolling(window=14, min_periods=14).sum().values
    
    # Chop = 100 * log10(sum(atr14) / (max(high14) - min(low14))) / log10(14)
    # Avoid division by zero
    price_range = high_14 - low_14
    chop_raw = np.full_like(close_1d, np.nan)
    mask = (price_range > 0) & (atr_14 > 0) & ~np.isnan(price_range) & ~np.isnan(atr_14)
    chop_raw[mask] = 100 * np.log10(atr_14[mask] / price_range[mask]) / np.log10(14)
    
    # Align Chop to 12h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    # Chop regimes: > 61.8 = ranging (mean revert), < 38.2 = trending
    chop_range = chop_1d_aligned > 61.8  # ranging market - good for mean reversion
    chop_trend = chop_1d_aligned < 38.2  # trending market - avoid
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(chop_1d_aligned[i]) or np.isnan(volume_20_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Only trade in ranging markets (chop > 61.8) where breakouts often fail and reverse
            if chop_range.iloc[i]:
                # Long breakout: price > R4 with volume spike
                if (prices['close'].iloc[i] > r4_1d_aligned[i] and 
                    vol_spike.iloc[i]):
                    position = 1
                    signals[i] = 0.25
                # Short breakdown: price < S4 with volume spike
                elif (prices['close'].iloc[i] < s4_1d_aligned[i] and 
                      vol_spike.iloc[i]):
                    position = -1
                    signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price retests R3/S3 (mean reversion signal in ranging market)
            # 2. Volume drops below average (loss of momentum)
            # 3. Market starts trending (chop < 38.2) - avoid fighting trend
            if position == 1:  # Long position
                if (prices['close'].iloc[i] < r3_1d_aligned[i] or 
                    vol_normal.iloc[i] or 
                    chop_trend.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (prices['close'].iloc[i] > s3_1d_aligned[i] or 
                    vol_normal.iloc[i] or 
                    chop_trend.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals
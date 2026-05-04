#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot Breakout + 1d Volume Spike + 1w Choppiness Regime
# Camarilla levels (R3, S3) from prior 1d provide high-probability reversal/breakout levels
# 1w Choppiness Index > 61.8 = ranging (fade extremes), < 38.2 = trending (breakout follow)
# 1d Volume > 2.0 x 20-period EMA confirms institutional participation
# In trending regime: breakout above R3 or below S3 with volume → follow trend
# In ranging regime: price at R3/S3 with volume → fade to mean (opposite direction)
# Designed for 12h timeframe targeting 50-150 total trades over 4 years (12-37/year).
# Works in bull/bear via regime adaptation: breakouts in trends, mean reversion in ranges.

name = "12h_Camarilla_Pivot_Breakout_1dVolumeSpike_1wChop"
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
    
    # Get 1d data for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for Choppiness Index regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on prior day OHLC)
    # R4 = Close + 1.5*(High-Low), R3 = Close + 1.125*(High-Low)
    # S3 = Close - 1.125*(High-Low), S4 = Close - 1.5*(High-Low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels use prior day's OHLC (shifted by 1 to avoid look-ahead)
    range_1d = high_1d - low_1d
    R3 = close_1d + 1.125 * range_1d
    S3 = close_1d - 1.125 * range_1d
    
    # Shift to align with current 12h bar (use prior completed 1d bar)
    R3 = np.concatenate([[np.nan], R3[:-1]])  # R3 from prior 1d bar
    S3 = np.concatenate([[np.nan], S3[:-1]])  # S3 from prior 1d bar
    
    # Calculate 1d volume EMA for spike detection
    vol_1d = df_1d['volume'].values
    vol_ema_20_1d = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1w Choppiness Index (CHOP)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range for 1w
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.maximum(np.maximum(tr1, tr2), tr3)
    tr_1w = np.concatenate([[np.nan], tr_1w])
    
    # ATR(14) for 1w
    atr_1w = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * LOG10(SUM(TR_14) / (HH14 - LL14)) / LOG10(14)
    tr_sum_14 = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_1w - ll_1w
    chop = np.where(range_14 != 0, 
                    100 * np.log10(tr_sum_14 / range_14) / np.log10(14), 
                    50)  # neutral when range=0
    
    # Align all HTF data to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    vol_ema_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_20_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # Calculate 12h ATR for stoploss
    tr_12h1 = np.abs(high[1:] - low[1:])
    tr_12h2 = np.abs(high[1:] - close[:-1])
    tr_12h3 = np.abs(low[1:] - close[:-1])
    tr_12h = np.maximum(np.maximum(tr_12h1, tr_12h2), tr_12h3)
    tr_12h = np.concatenate([[np.nan], tr_12h])
    atr_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(vol_ema_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(atr_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 2.0 x 20-period 1d EMA
        volume_confirm = volume[i] > (2.0 * vol_ema_aligned[i])
        
        # Regime filter: 1w Choppiness Index
        chop_val = chop_aligned[i]
        trending_regime = chop_val < 38.2   # CHOP < 38.2 = trending
        ranging_regime = chop_val > 61.8    # CHOP > 61.8 = ranging
        
        if position == 0:
            if volume_confirm:
                if trending_regime:
                    # Trending: breakout follow
                    if close[i] > R3_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                    elif close[i] < S3_aligned[i]:
                        signals[i] = -0.25
                        position = -1
                elif ranging_regime:
                    # Ranging: fade extremes at pivot levels
                    if close[i] >= R3_aligned[i] * 0.999:  # near R3 with tolerance
                        signals[i] = -0.25  # short at resistance
                        position = -1
                    elif close[i] <= S3_aligned[i] * 1.001:  # near S3 with tolerance
                        signals[i] = 0.25   # long at support
                        position = 1
        elif position == 1:
            # Exit long: close below S3 OR chop shifts to extreme ranging
            if close[i] < S3_aligned[i] or chop_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: close above R3 OR chop shifts to extreme ranging
            if close[i] > R3_aligned[i] or chop_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
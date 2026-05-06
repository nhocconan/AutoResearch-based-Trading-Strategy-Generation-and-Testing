#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Trix(12) + Volume Spike + Choppiness Regime Filter
# Uses TRIX (12-period) to detect momentum shifts in 12h timeframe
# Volume confirmation (>2x 20-bar avg) ensures breakout strength
# Choppiness filter: CHOP(14) > 61.8 = range (mean reversion), CHOP < 38.2 = trending (follow momentum)
# Works in both bull/bear: TRIX captures momentum, volume filter avoids false breakouts, chop filter adapts to regime
# Discrete sizing 0.25 to balance profit and fee drag; target 50-150 total trades over 4 years (12-37/year)

name = "12h_Trix12_VolumeChop_v1"
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
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Choppiness Index (14-period)
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    chop_raw = 100 * np.log10(atr_1d.sum() / (highest_high_1d - lowest_low_1d))
    chop = pd.Series(chop_raw).rolling(window=14, min_periods=14).mean().values
    
    # Calculate TRIX(12) on 12h close
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 15:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # TRIX: EMA(EMA(EMA(close, 12), 12), 12) - 1, then * 100
    ema1 = pd.Series(close_12h).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = (ema3 / ema3.shift(1) - 1) * 100
    trix = trix.fillna(0).values  # Handle first value
    
    # Calculate volume spike filter (>2.0x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_20)
    
    # Align HTF indicators to 12h timeframe (primary)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    trix_aligned = align_htf_to_ltf(prices, df_12h, trix)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(trix_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX > 0 (bullish momentum) AND chop < 38.2 (trending) AND volume spike
            if trix_aligned[i] > 0 and chop_aligned[i] < 38.2 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX < 0 (bearish momentum) AND chop < 38.2 (trending) AND volume spike
            elif trix_aligned[i] < 0 and chop_aligned[i] < 38.2 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX turns negative OR chop > 61.8 (range) OR volume drops
            if trix_aligned[i] <= 0 or chop_aligned[i] > 61.8 or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX turns positive OR chop > 61.8 (range) OR volume drops
            if trix_aligned[i] >= 0 or chop_aligned[i] > 61.8 or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
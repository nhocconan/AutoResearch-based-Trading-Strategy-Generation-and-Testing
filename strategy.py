#!/usr/bin/env python3
"""
1d_1w_Camarilla_R1_S1_Breakout_Volume_Regime_v1
Concept: Daily Camarilla R1/S1 breakout with weekly trend filter, volume confirmation, and chop regime filter.
- Long: Price breaks above R1 AND weekly EMA10 > EMA20 (uptrend) AND volume > 1.5x avg AND chop < 61.8 (trending)
- Short: Price breaks below S1 AND weekly EMA10 < EMA20 (downtrend) AND volume > 1.5x avg AND chop < 61.8 (trending)
- Exit: Price crosses back below R1 (long) or above S1 (short)
- Position sizing: 0.25
- Target: 10-25 trades/year (40-100 total over 4 years)
- Works in bull/bear: Weekly EMA defines trend, Camarilla provides precise levels, volume/chop filters avoid false breaks
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_R1_S1_Breakout_Volume_Regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === Daily: Calculate Camarilla levels (using previous day's OHLC) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Shift to get previous day's values
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # First day has no previous day
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan
    
    # Calculate pivot and range
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels
    R1 = pivot + (range_val * 1.0 / 12)
    S1 = pivot - (range_val * 1.0 / 12)
    
    # === Weekly: EMA10 and EMA20 for trend filter ===
    weekly_close = df_1w['close'].values
    ema10_w = pd.Series(weekly_close).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema20_w = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly EMAs to daily
    ema10_w_aligned = align_htf_to_ltf(prices, df_1w, ema10_w)
    ema20_w_aligned = align_htf_to_ltf(prices, df_1w, ema20_w)
    
    # === Daily: Volume filter (1.5x 20-day average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma20 * 1.5
    
    # === Daily: Chop filter (choppiness index < 61.8 = trending) ===
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Calculate highest high and lowest low over ATR period
    hh = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    ll = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Chop calculation: 100 * log10(sum(TR/atr) / (hh - ll)) / log10(atr_period)
    sum_tr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).sum().values
    # Avoid division by zero
    denominator = hh - ll
    denominator = np.where(denominator == 0, 1e-10, denominator)
    chop = 100 * (np.log10(sum_tr / atr) / np.log10(atr_period))
    chop = np.where(denominator == 0, 50.0, chop)  # Neutral when range is zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        r1_val = R1[i]
        s1_val = S1[i]
        close_val = close[i]
        ema10_val = ema10_w_aligned[i]
        ema20_val = ema20_w_aligned[i]
        vol_val = volume[i]
        vol_thresh = vol_threshold[i]
        chop_val = chop[i]
        
        # Skip if any value is NaN
        if (np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(close_val) or 
            np.isnan(ema10_val) or np.isnan(ema20_val) or np.isnan(vol_val) or 
            np.isnan(vol_thresh) or np.isnan(chop_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 AND weekly uptrend AND high volume AND trending market
            if (close_val > r1_val and ema10_val > ema20_val and 
                vol_val > vol_thresh and chop_val < 61.8):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 AND weekly downtrend AND high volume AND trending market
            elif (close_val < s1_val and ema10_val < ema20_val and 
                  vol_val > vol_thresh and chop_val < 61.8):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses back below R1
            if close_val < r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses back above S1
            if close_val > s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
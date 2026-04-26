#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_ChopRegime_ATRStop_v3
Hypothesis: Combines Camarilla R3/S3 breakouts with 1d trend filter, choppiness regime (CHOP > 61.8 = range, < 38.2 = trend), and ATR-based trailing stop. Uses discrete sizing (±0.25) to minimize fee churn. Long when price breaks above R3 in bullish 1d trend AND low chop (trending); short when breaks below S3 in bearish 1d trend AND low chop. Exits on trend reversal, chop increase, or ATR trailing stop hit. Designed for fewer trades (<30/year) and better bear market performance via regime adaptation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for higher-timeframe trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate previous 1d bar's Camarilla levels (using 1d data)
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous 1d bar's high, low, close for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values  # Shift to get previous 1d bar
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_range = prev_high - prev_low
    r3 = prev_close + camarilla_range * 1.1 / 4
    s3 = prev_close - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Choppiness Index regime filter: CHOP > 61.8 = range, CHOP < 38.2 = trending
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # ATR(14) for denominator
    atr_period = 14
    atr = np.zeros(n)
    atr[atr_period] = np.mean(tr[1:atr_period+1])
    for i in range(atr_period+1, n):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Sum of TR over CHOP period (14)
    chop_period = 14
    tr_sum = np.zeros(n)
    for i in range(chop_period, n):
        tr_sum[i] = np.sum(tr[i-chop_period+1:i+1])
    
    # Highest high and lowest low over CHOP period
    max_high = np.zeros(n)
    min_low = np.zeros(n)
    for i in range(chop_period, n):
        max_high[i] = np.max(high[i-chop_period+1:i+1])
        min_low[i] = np.min(low[i-chop_period+1:i+1])
    
    # Choppiness Index: CHOP = 100 * log10(tr_sum / (max_high - min_low)) / log10(chop_period)
    # Avoid division by zero and log of zero
    range_hl = max_high - min_low
    chop = np.zeros(n)
    for i in range(chop_period, n):
        if range_hl[i] > 0 and tr_sum[i] > 0:
            chop[i] = 100 * np.log10(tr_sum[i] / range_hl[i]) / np.log10(chop_period)
        else:
            chop[i] = 50  # Neutral value
    
    # Regime: trending when CHOP < 38.2, ranging when CHOP > 61.8
    trending_regime = chop < 38.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Track highest high since entry for long, lowest low for short (for ATR trailing stop)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    # ATR multiplier for trailing stop
    atr_multiplier = 3.0
    
    # Warmup: max of calculations (20 for volume MA, 1 for shift, 34 for EMA, 14 for ATR/CHOP)
    start_idx = max(20, 1, 34, chop_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready (NaN from calculation)
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(chop[i]) or np.isnan(atr[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        chop_val = chop[i]
        atr_val = atr[i]
        trending_regime_now = trending_regime[i]
        
        # Update highest/lowest since entry
        if position == 1:
            highest_since_entry[i] = max(highest_since_entry[i-1] if i > 0 else close_val, close_val)
            lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else close_val
        elif position == -1:
            highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else close_val
            lowest_since_entry[i] = min(lowest_since_entry[i-1] if i > 0 else close_val, close_val)
        else:
            highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else close_val
            lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else close_val
        
        # Determine 1d trend: bullish if price > EMA34, bearish if price < EMA34
        bullish_1d = close_val > ema_34_val
        bearish_1d = close_val < ema_34_val
        
        # ATR trailing stop levels
        long_stop = highest_since_entry[i] - (atr_multiplier * atr_val) if position == 1 else 0
        short_stop = lowest_since_entry[i] + (atr_multiplier * atr_val) if position == -1 else 0
        
        # Check if ATR trailing stop is hit
        long_stop_hit = position == 1 and close_val < long_stop
        short_stop_hit = position == -1 and close_val > short_stop
        
        # Entry conditions: price breaks above/below Camarilla levels in direction of 1d trend with volume confirmation and trending regime (low chop)
        long_entry = (close_val > r3_val) and bullish_1d and vol_spike and trending_regime_now
        short_entry = (close_val < s3_val) and bearish_1d and vol_spike and trending_regime_now
        
        # Exit conditions: 
        # 1. Price returns inside Camarilla levels
        # 2. Trend reversal (1d trend changes)
        # 3. Regime change to ranging (chop increases)
        # 4. ATR trailing stop hit
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and (close_val < r3_val or not bullish_1d or not trending_regime_now or long_stop_hit):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close_val > s3_val or not bearish_1d or not trending_regime_now or short_stop_hit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_ChopRegime_ATRStop_v3"
timeframe = "4h"
leverage = 1.0
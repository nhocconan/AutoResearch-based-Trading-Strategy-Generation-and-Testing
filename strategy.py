#!/usr/bin/env python3
"""
12h_Williams_Alligator_1wTrend_HTFRegime_v1
Hypothesis: Trade 12h Williams Alligator signals aligned with 1w trend and choppiness regime filter. 
Williams Alligator (Jaw=13, Teeth=8, Lips=5) provides trend direction: 
- Bullish when Lips > Teeth > Jaw (alligator mouth open upward) 
- Bearish when Lips < Teeth < Jaw (alligator mouth open downward)
Uses 1w EMA50 as higher timeframe trend filter to avoid counter-trend trades. 
Adds 1d Choppiness Index regime filter: only trade when CHOP(14) > 61.8 (ranging market) 
to catch mean-reversion swings within the Alligator's teeth during low volatility.
Exit on Alligator reversal or opposing Camarilla touch for risk control.
Position size: 0.25 discrete to minimize fee churn. 
Target: 80-120 total trades over 4 years = 20-30/year (within 12h optimal range).
Uses 1w HTF for more stable trend alignment than 1d, which should improve performance in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for HTF trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Choppiness Index regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d Choppiness Index (CHOP)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # ATR(14) and sum of true ranges
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sum_TR_14 / (ATR_14 * 14)) / log10(14)
    # Avoid division by zero
    chop_denominator = atr_14 * 14
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)
    chop_ratio = sum_tr_14 / chop_denominator
    chop_ratio = np.where(chop_ratio <= 0, 1e-10, chop_ratio)  # avoid log10(<=0)
    chop = 100 * np.log10(chop_ratio) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Get 12h data for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 12h data
    # Jaw (blue line): 13-period SMMA smoothed 8 periods ahead
    # Teeth (red line): 8-period SMMA smoothed 5 periods ahead  
    # Lips (green line): 5-period SMMA smoothed 3 periods ahead
    close_12h = df_12h['close'].values
    
    # SMMA (Smoothed Moving Average) calculation
    def smma(source, period):
        if len(source) < period:
            return np.full_like(source, np.nan)
        result = np.full_like(source, np.nan, dtype=np.float64)
        # First value is SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + PRICE) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw_raw = smma(close_12h, 13)
    teeth_raw = smma(close_12h, 8)
    lips_raw = smma(close_12h, 5)
    
    # Apply smoothing offsets
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Align Alligator lines to 12h prices
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Get 12h Camarilla levels for stop loss
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    hl_range_12h = high_12h - low_12h
    # Camarilla R3 and S3 (stronger stop levels)
    r3_12h = close_12h + (1.1 * hl_range_12h / 4)  # R3 = close + 1.1*(high-low)/4
    s3_12h = close_12h - (1.1 * hl_range_12h / 4)  # S3 = close - 1.1*(high-low)/4
    
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Alligator (13) and CHOP (14)
    start_idx = 13
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(chop_aligned[i]) or
            np.isnan(jaw_aligned[i]) or
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1w HTF trend (bullish = price above 1w EMA50)
        htf_1w_bullish = close[i] > ema_50_1w_aligned[i]
        htf_1w_bearish = close[i] < ema_50_1w_aligned[i]
        
        # Regime filter: only trade in ranging markets (CHOP > 61.8)
        ranging_market = chop_aligned[i] > 61.8
        
        # Williams Alligator signals
        # Bullish: Lips > Teeth > Jaw (mouth open upward)
        alligator_bullish = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        # Bearish: Lips < Teeth < Jaw (mouth open downward)
        alligator_bearish = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        if position == 0:
            # Long setup: Alligator bullish + 1w uptrend + ranging regime
            long_setup = alligator_bullish and htf_1w_bullish and ranging_market
            
            # Short setup: Alligator bearish + 1w downtrend + ranging regime
            short_setup = alligator_bearish and htf_1w_bearish and ranging_market
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: Alligator turns bearish OR price touches 12h Camarilla S3 (stop)
            if alligator_bearish or (close[i] <= s3_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: Alligator turns bullish OR price touches 12h Camarilla R3 (stop)
            if alligator_bullish or (close[i] >= r3_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Williams_Alligator_1wTrend_HTFRegime_v1"
timeframe = "12h"
leverage = 1.0
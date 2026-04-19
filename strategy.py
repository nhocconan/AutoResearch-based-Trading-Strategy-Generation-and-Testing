#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator (Jaw/Teeth/Lips) with 1w trend filter.
# Uses smoothed SMAs (13,8,5 periods) to identify trend direction and strength.
# Long when Lips > Teeth > Jaw (bullish alignment), short when reverse.
# Weekly trend filter ensures alignment with higher timeframe direction.
# Target: 50-150 total trades over 4 years (12-37/year) with low churn.
name = "6h_Alligator_1wTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate Williams Alligator components (13,8,5 smoothed SMAs)
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward  
    # Lips: 5-period SMMA, shifted 3 bars forward
    def smoothed_sma(data, period):
        # Smoothed Moving Average (SMMA) - Wilder's smoothing
        sma = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            sma[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                sma[i] = (sma[i-1] * (period-1) + data[i]) / period
        return sma
    
    jaw_raw = smoothed_sma(close, 13)
    teeth_raw = smoothed_sma(close, 8)
    lips_raw = smoothed_sma(close, 5)
    
    # Apply shifts (Alligator specific)
    jaw = np.roll(jaw_raw, 8)  # shift 8 bars forward
    teeth = np.roll(teeth_raw, 5)  # shift 5 bars forward
    lips = np.roll(lips_raw, 3)  # shift 3 bars forward
    
    # Weekly trend filter: EMA34 on weekly close
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Additional delay for weekly EMA (needs confirmation)
    # For EMA, 1-bar delay from align_htf_to_ltf is sufficient
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after sufficient data for Alligator calculation
    start_idx = max(13 + 8, 34)  # Jaw needs 13+8=21, plus weekly EMA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment conditions
        lips_gt_teeth = lips[i] > teeth[i]
        teeth_gt_jaw = teeth[i] > jaw[i]
        bullish_alignment = lips_gt_teeth and teeth_gt_jaw
        
        lips_lt_teeth = lips[i] < teeth[i]
        teeth_lt_jaw = teeth[i] < jaw[i]
        bearish_alignment = lips_lt_teeth and teeth_lt_jaw
        
        # Weekly trend filter
        price_vs_weekly_ema = close[i] > ema_34_1w_aligned[i]
        
        if position == 0:
            # Enter long: bullish alignment AND price above weekly EMA (uptrend)
            if bullish_alignment and price_vs_weekly_ema:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish alignment AND price below weekly EMA (downtrend)
            elif bearish_alignment and not price_vs_weekly_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish alignment or price crosses below weekly EMA
            if bearish_alignment or not price_vs_weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish alignment or price crosses above weekly EMA
            if bullish_alignment or price_vs_weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
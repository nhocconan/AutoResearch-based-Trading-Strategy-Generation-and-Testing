#!/usr/bin/env python3
"""
1d_WilliamsAlligator_1wTrend_Filter
Hypothesis: Uses Williams Alligator (3 SMAs: Jaw=13, Teeth=8, Lips=5) to detect trend direction on daily timeframe, filtered by weekly trend (EMA50) to trade only in higher timeframe direction. Alligator signals when Lips > Teeth > Jaw (bullish) or Lips < Teeth < Jaw (bearish). Weekly EMA50 filter ensures alignment with long-term trend, reducing whipsaws in ranging markets. Designed for low trade frequency (10-25/year) to minimize fee decay on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams Alligator components on daily data
    # Jaw = SMMA(13, 8) - 13-period smoothed moving average, shifted 8 bars
    # Teeth = SMMA(8, 5) - 8-period smoothed moving average, shifted 5 bars  
    # Lips = SMMA(5, 3) - 5-period smoothed moving average, shifted 3 bars
    
    def smma(series, period, shift):
        """Smoothed Moving Average (SMMA) with shift"""
        sma = pd.Series(series).rolling(window=period, min_periods=period).mean().values
        # SMMA calculation: (prev_smma * (period-1) + current_price) / period
        smma_vals = np.full_like(series, np.nan)
        if len(series) >= period:
            smma_vals[period-1] = sma[period-1]  # First value is SMA
            for i in range(period, len(series)):
                if not np.isnan(smma_vals[i-1]):
                    smma_vals[i] = (smma_vals[i-1] * (period-1) + series[i]) / period
                else:
                    smma_vals[i] = np.nan
        # Apply shift
        shifted = np.full_like(series, np.nan)
        if shift < len(series):
            shifted[shift:] = smma_vals[:-shift]
        return shifted
    
    jaw = smma(close, 13, 8)
    teeth = smma(close, 8, 5)
    lips = smma(close, 5, 3)
    
    # Align weekly EMA50 to daily timeframe
    # (already done above)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for Alligator to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Alligator signals: alignment of Lips, Teeth, Jaw
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_50_1w_aligned[i]
        weekly_downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Entry conditions: Alligator alignment with weekly trend
        long_entry = bullish_alignment and weekly_uptrend
        short_entry = bearish_alignment and weekly_downtrend
        
        # Exit conditions: Alligator loses alignment or opposite signal
        long_exit = not bullish_alignment  # Any misalignment exits long
        short_exit = not bearish_alignment  # Any misalignment exits short
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_WilliamsAlligator_1wTrend_Filter"
timeframe = "1d"
leverage = 1.0
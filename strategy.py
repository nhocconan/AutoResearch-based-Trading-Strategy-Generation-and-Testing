# 12h_Williams_Alligator_Trend_Confirmation_v1
# Hypothesis: Uses Williams Alligator (Jaw/Teeth/Lips) on 12h chart with 1d/200 EMA trend filter.
# Long when price > Teeth and Lips > Jaw (bullish alignment) and price > 1d EMA200.
# Short when price < Teeth and Lips < Jaw (bearish alignment) and price < 1d EMA200.
# Williams Alligator uses smoothed moving averages (SMMA) to filter noise.
# Designed for low trade frequency (<25/year) to avoid fee drag while capturing trends.
# Works in both bull and bear markets by following the trend defined by higher timeframe.

name = "12h_Williams_Alligator_Trend_Confirmation_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(data, period):
    """Smoothed Moving Average (SMMA)"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    result = np.full_like(data, np.nan, dtype=float)
    # First value is SMA
    result[period-1] = np.mean(data[:period])
    # Subsequent values: SMMA = (prev_smma * (period-1) + current_price) / period
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get 1d data for 200 EMA trend filter (updated daily)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # --- 1d EMA200 for trend filter ---
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # --- Williams Alligator on 12h data (Jaw=13, Teeth=8, Lips=5) ---
    # All lines are SMMA (Smoothed Moving Average)
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Align Alligator lines (already on 12h, no alignment needed for same TF)
    # But we'll keep the pattern for consistency if we ever change TF
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Alligator)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(jaw[i]) or
            np.isnan(teeth[i]) or
            np.isnan(lips[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Williams Alligator signals:
        # Bullish: Lips > Teeth > Jaw (all aligned upward)
        # Bearish: Lips < Teeth < Jaw (all aligned downward)
        bullish_alignment = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        bearish_alignment = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        # Trend filter: price vs 1d EMA200
        price_above_ema200 = close[i] > ema_200_1d_aligned[i]
        price_below_ema200 = close[i] < ema_200_1d_aligned[i]
        
        if position == 0:
            # Look for new entries
            if bullish_alignment and price_above_ema200:
                # Strong bullish alignment with higher timeframe uptrend
                signals[i] = 0.25
                position = 1
            elif bearish_alignment and price_below_ema200:
                # Strong bearish alignment with higher timeframe downtrend
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: reverse Alligator alignment or trend change
            if position == 1:
                # Exit long: bearish alignment OR price breaks below EMA200
                if bearish_alignment or price_below_ema200:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: bullish alignment OR price breaks above EMA200
                if bullish_alignment or price_above_ema200:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
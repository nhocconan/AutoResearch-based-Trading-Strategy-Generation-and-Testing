#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly Bollinger Band Squeeze Breakout with Volume Confirmation
# - Identify weekly BB squeeze (BB width < 20-period percentile 20)
# - Long when price breaks above upper BB with volume > 1.5x 20-day average
# - Short when price breaks below lower BB with volume > 1.5x 20-day average
# - Exit when price re-enters BB bands or squeeze ends
# - Uses weekly timeframe for structure, daily for entry/exit
# - Target: 7-25 trades per year per symbol (30-100 total over 4 years)
# - Designed to work in both bull and breakout markets

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for BB calculation
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly Bollinger Bands (20, 2.0)
    close_ser = pd.Series(close_weekly)
    basis = close_ser.rolling(window=20, min_periods=20).mean()
    dev = close_ser.rolling(window=20, min_periods=20).std()
    upper = basis + 2.0 * dev
    lower = basis - 2.0 * dev
    bb_width = upper - lower
    
    # Calculate BB width percentile (20-period) for squeeze detection
    bb_width_ser = pd.Series(bb_width.values)
    width_percentile = bb_width_ser.rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Squeeze condition: BB width < 20th percentile
    squeeze = width_percentile < 20.0
    
    # Align weekly indicators to daily timeframe
    upper_aligned = align_htf_to_ltf(prices, df_weekly, upper.values)
    lower_aligned = align_htf_to_ltf(prices, df_weekly, lower.values)
    squeeze_aligned = align_htf_to_ltf(prices, df_weekly, squeeze)
    
    # Daily volume average (20-period)
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in indicators
        if np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(squeeze_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        
        if position == 0:
            # Look for breakout after squeeze
            if squeeze_aligned[i] and vol > 1.5 * vol_ma[i]:
                if price > upper_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif price < lower_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: price re-enters BB or squeeze ends
            if price < upper_aligned[i] or not squeeze_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price re-enters BB or squeeze ends
            if price > lower_aligned[i] or not squeeze_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyBB_SqueezeBreakout_Volume"
timeframe = "1d"
leverage = 1.0
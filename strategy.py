#!/usr/bin/env python3
"""
12h_WilliamsAlligator_1wTrend_Volume
Hypothesis: Williams Alligator (jaw/teeth/lips) on 12h filters direction, with 1w EMA50 trend filter and volume spike confirmation.
Only trade when price is outside the Alligator's mouth (lips outside jaw/teeth) in the direction of the 1w trend.
Volume must be above 1.5x median of last 50 periods to confirm conviction.
Designed for 12-30 trades/year per symbol to minimize fee drag while capturing strong trends.
Works in both bull and bear markets due to strong trend filter and volume confirmation.
"""

name = "12h_WilliamsAlligator_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def williams_alligator(close, jaw_period=13, teeth_period=8, lips_period=5,
                       jaw_shift=8, teeth_shift=5, lips_shift=3):
    """Calculate Williams Alligator lines (SMMA based)."""
    # SMMA (Smoothed Moving Average) approximation using EMA for simplicity
    # In practice, SMMA = (prev_smma * (n-1) + close) / n
    # We'll use EMA as proxy which is commonly accepted
    jaw = pd.Series(close).ewm(span=jaw_period, adjust=False).mean().values
    teeth = pd.Series(close).ewm(span=teeth_period, adjust=False).mean().values
    lips = pd.Series(close).ewm(span=lips_period, adjust=False).mean().values
    
    # Apply shifts (delay)
    jaw = np.roll(jaw, jaw_shift)
    teeth = np.roll(teeth, teeth_shift)
    lips = np.roll(lips, lips_shift)
    
    # Fill NaN from roll with first valid value
    jaw[:jaw_shift] = jaw[jaw_shift] if jaw_shift < len(jaw) else 0
    teeth[:teeth_shift] = teeth[teeth_shift] if teeth_shift < len(teeth) else 0
    lips[:lips_shift] = lips[lips_shift] if lips_shift < len(lips) else 0
    
    return jaw, teeth, lips

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    volume_12h = prices['volume'].values
    
    # --- 1w Trend Filter: EMA50 ---
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # --- 12h Williams Alligator (13,8,5) ---
    jaw, teeth, lips = williams_alligator(close_12h, 13, 8, 5, 8, 5, 3)
    
    # --- Volume Filter: spike above 1.5x median of last 50 periods ---
    vol_median = pd.Series(volume_12h).rolling(window=50, min_periods=20).median().values
    vol_threshold = vol_median * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 60  # for Alligator and EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_threshold[i])):
            if position != 0:
                # Check stoploss (using ATR approximation from high-low)
                # Approximate ATR as 20-period average of (high-low)
                if i >= 20:
                    atr_approx = np.mean(high_12h[i-20:i] - low_12h[i-20:i])
                else:
                    atr_approx = np.mean(high_12h[:i+1] - low_12h[:i+1])
                if position == 1 and close_12h[i] <= entry_price - 2.0 * atr_approx:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_12h[i] >= entry_price + 2.0 * atr_approx:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine 1w trend
        trend_up = close_12h[i] > ema50_1w_aligned[i]
        trend_down = close_12h[i] < ema50_1w_aligned[i]
        
        # Alligator conditions: lips outside jaw/teeth in direction of trend
        # For long: lips > jaw AND lips > teeth (bullish alignment)
        # For short: lips < jaw AND lips < teeth (bearish alignment)
        lips_above_jaw = lips[i] > jaw[i]
        lips_above_teeth = lips[i] > teeth[i]
        lips_below_jaw = lips[i] < jaw[i]
        lips_below_teeth = lips[i] < teeth[i]
        
        # Volume filter: spike above 1.5x median
        vol_ok = volume_12h[i] > vol_threshold[i]
        
        if position == 0:
            # Look for entries only in direction of 1w trend with Alligator alignment and volume spike
            if trend_up and lips_above_jaw and lips_above_teeth and vol_ok:
                # Long: price above Alligator teeth (strong bullish) + 1w uptrend + volume spike
                signals[i] = 0.25
                position = 1
                entry_price = close_12h[i]
            elif trend_down and lips_below_jaw and lips_below_teeth and vol_ok:
                # Short: price below Alligator teeth (strong bearish) + 1w downtrend + volume spike
                signals[i] = -0.25
                position = -1
                entry_price = close_12h[i]
        else:
            # Exit conditions: Alligator lines re-cross (trend weakening) or stoploss
            if position == 1:
                # Stoploss approximation
                if i >= 20:
                    atr_approx = np.mean(high_12h[i-20:i] - low_12h[i-20:i])
                else:
                    atr_approx = np.mean(high_12h[:i+1] - low_12h[:i+1])
                if close_12h[i] <= entry_price - 2.0 * atr_approx:
                    signals[i] = 0.0
                    position = 0
                # Exit: lips cross back below teeth (weakening bullish momentum)
                elif lips[i] < teeth[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Stoploss approximation
                if i >= 20:
                    atr_approx = np.mean(high_12h[i-20:i] - low_12h[i-20:i])
                else:
                    atr_approx = np.mean(high_12h[:i+1] - low_12h[:i+1])
                if close_12h[i] >= entry_price + 2.0 * atr_approx:
                    signals[i] = 0.0
                    position = 0
                # Exit: lips cross back above teeth (weakening bearish momentum)
                elif lips[i] > teeth[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
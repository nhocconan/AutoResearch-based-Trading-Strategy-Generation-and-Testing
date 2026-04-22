#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 123 Reversal with 12h Trend Filter and Volume Confirmation
# The 123 Reversal pattern identifies trend exhaustion and potential reversal points.
# Step 1: Price makes a new high/low. Step 2: Pullback forms a swing point. 
# Step 3: Price breaks through the swing point in opposite direction = reversal signal.
# Combined with 12h EMA50 trend filter to avoid counter-trend trades.
# Volume > 1.3x average confirms reversal strength.
# Works in both bull and bear markets by capturing exhaustion moves.
# Uses discrete position sizing (0.25) to minimize fee churn.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for EMA trend (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Swing points for 123 pattern
    # Swing high: high > previous high and next high
    # Swing low: low < previous low and next low
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    
    for i in range(2, n-2):
        # Swing high condition
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            swing_high[i] = True
        # Swing low condition
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            swing_low[i] = True
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    last_swing_high_idx = -1
    last_swing_low_idx = -1
    
    for i in range(5, n):
        # Skip if data not ready
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Update swing point indices
        if swing_high[i]:
            last_swing_high_idx = i
        if swing_low[i]:
            last_swing_low_idx = i
        
        if position == 0:
            # Long setup: 123 bullish reversal
            # Need a swing low, then price breaks above it
            if last_swing_low_idx != -1 and i - last_swing_low_idx <= 10:  # Within last 10 bars
                swing_low_price = low[last_swing_low_idx]
                # Condition 1: Price made new low (swing low)
                # Condition 2: Pullback formed (implicit by swing low)
                # Condition 3: Price breaks above swing low = reversal signal
                if close[i] > swing_low_price and close[i-1] <= swing_low_price:
                    # Additional filters: above 12h EMA (uptrend) and volume spike
                    if close[i] > ema_50_12h_aligned[i] and volume[i] > 1.3 * vol_avg_20[i]:
                        signals[i] = 0.25
                        position = 1
                        last_swing_low_idx = -1  # Reset after signal
            
            # Short setup: 123 bearish reversal
            # Need a swing high, then price breaks below it
            elif last_swing_high_idx != -1 and i - last_swing_high_idx <= 10:  # Within last 10 bars
                swing_high_price = high[last_swing_high_idx]
                # Condition 1: Price made new high (swing high)
                # Condition 2: Pullback formed (implicit by swing high)
                # Condition 3: Price breaks below swing high = reversal signal
                if close[i] < swing_high_price and close[i-1] >= swing_high_price:
                    # Additional filters: below 12h EMA (downtrend) and volume spike
                    if close[i] < ema_50_12h_aligned[i] and volume[i] > 1.3 * vol_avg_20[i]:
                        signals[i] = -0.25
                        position = -1
                        last_swing_high_idx = -1  # Reset after signal
        else:
            # Exit conditions: reversal in opposite direction or trend change
            if position == 1:
                # Exit long: bearish 123 reversal or price below 12h EMA
                bearish_reversal = False
                if last_swing_high_idx != -1 and i - last_swing_high_idx <= 10:
                    swing_high_price = high[last_swing_high_idx]
                    if close[i] < swing_high_price and close[i-1] >= swing_high_price:
                        bearish_reversal = True
                
                if bearish_reversal or close[i] < ema_50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: bullish 123 reversal or price above 12h EMA
                bullish_reversal = False
                if last_swing_low_idx != -1 and i - last_swing_low_idx <= 10:
                    swing_low_price = low[last_swing_low_idx]
                    if close[i] > swing_low_price and close[i-1] <= swing_low_price:
                        bullish_reversal = True
                
                if bullish_reversal or close[i] > ema_50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_123Reversal_12hEMA50_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume spike confirmation
# Williams Alligator uses three smoothed moving averages (Jaws, Teeth, Lips) to identify trends.
# When the three lines are intertwined (no clear trend), we stay out (choppy market).
# When they diverge in order (Lips > Teeth > Jaws for uptrend, reverse for downtrend), we trend follow.
# 1d EMA50 filter ensures alignment with daily trend for higher probability trades.
# Volume spike (>1.5x 20-period average) confirms institutional participation.
# Designed for 4h timeframe targeting 20-40 trades/year with strong performance in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Williams Alligator: three SMMA (Smoothed Moving Average) lines
    # Jaws: SMMA(13, 8) - blue line
    # Teeth: SMMA(8, 5) - red line  
    # Lips: SMMA(5, 3) - green line
    
    def smoothed_moving_average(data, period, shift):
        """Calculate Smoothed Moving Average (SMMA)"""
        sma = pd.Series(data).rolling(window=period, min_periods=period).mean().values
        # SMMA is like EMA but with specific smoothing - we'll use EMA as approximation
        # For Williams Alligator, SMMA period is typically calculated with specific smoothing
        alpha = 2.0 / (period + 1)
        smoothed = np.full_like(data, np.nan, dtype=float)
        smoothed[period-1] = sma[period-1]  # Start with SMA value
        for i in range(period, len(data)):
            if not np.isnan(smoothed[i-1]) and not np.isnan(data[i]):
                smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
            else:
                smoothed[i] = smoothed[i-1]
        # Apply shift
        shifted = np.full_like(data, np.nan, dtype=float)
        if shift < len(smoothed):
            shifted[shift:] = smoothed[:-shift] if shift > 0 else smoothed
        return shifted
    
    # Calculate Alligator lines
    jaws = smoothed_moving_average(close, 13, 8)   # SMMA(13, 8)
    teeth = smoothed_moving_average(close, 8, 5)   # SMMA(8, 5)
    lips = smoothed_moving_average(close, 5, 3)    # SMMA(5, 3)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaws (bullish alignment) + daily uptrend + volume spike
            if (lips[i] > teeth[i] and teeth[i] > jaws[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Jaws > Teeth > Lips (bearish alignment) + daily downtrend + volume spike
            elif (jaws[i] > teeth[i] and teeth[i] > lips[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: when Alligator lines re-intertwine (market goes choppy) or trend reversal
            if position == 1:
                # Exit long: when Teeth <= Jaws or Lips <= Teeth (loss of bullish alignment) or trend turns down
                if (teeth[i] <= jaws[i] or lips[i] <= teeth[i] or 
                    close[i] < ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: when Teeth >= Jaws or Lips >= Teeth (loss of bearish alignment) or trend turns up
                if (teeth[i] >= jaws[i] or lips[i] >= teeth[i] or 
                    close[i] > ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0
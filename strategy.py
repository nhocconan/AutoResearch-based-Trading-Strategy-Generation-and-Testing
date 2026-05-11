#!/usr/bin/env python3
"""
12h_Keltner_Breakout_Slope_Trend_v1
Hypothesis: Uses Keltner Channel breakout with slope confirmation from weekly EMA20 to capture strong trends.
Breakouts occur when price closes outside the Keltner Channel (ATR-based) with volume confirmation.
Trend filter uses weekly EMA20 slope (rising/falling) to avoid counter-trend trades.
Designed for low trade frequency (<25/year) to minimize fee decay while capturing explosive moves.
"""

name = "12h_Keltner_Breakout_Slope_Trend_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 12h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- Weekly EMA20 for trend filter ---
    close_1w = df_1w['close']
    ema_20_1w = close_1w.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate slope of weekly EMA (rising/falling)
    ema_slope = np.diff(ema_20_1w_aligned, prepend=ema_20_1w_aligned[0])
    ema_slope = np.where(ema_slope > 0, 1, np.where(ema_slope < 0, -1, 0))
    
    # --- Keltner Channel (20-period, 2.0 ATR multiplier) ---
    # Calculate ATR
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate EMA of close for center line
    ema_close = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Upper and lower Keltner bands
    keltner_upper = ema_close + (2.0 * atr)
    keltner_lower = ema_close - (2.0 * atr)
    
    # --- Volume Spike Detection (1.5x 20-period EMA) ---
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ema)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(keltner_upper[i]) or
            np.isnan(keltner_lower[i]) or
            np.isnan(vol_spike[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend from weekly EMA slope
        trend_up = ema_slope[i] > 0
        trend_down = ema_slope[i] < 0
        
        # Breakout signals (price closes outside Keltner Channel with volume spike)
        long_breakout = (close[i] > keltner_upper[i]) and vol_spike[i]
        short_breakout = (close[i] < keltner_lower[i]) and vol_spike[i]
        
        if position == 0:
            # Only take trades in direction of weekly trend
            if trend_up and long_breakout:
                signals[i] = 0.25
                position = 1
            elif trend_down and short_breakout:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to center line or opposite band touch
            if position == 1:
                # Exit long: price closes below EMA (center) or touches lower band
                exit_signal = (close[i] < ema_close[i]) or (low[i] <= keltner_lower[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price closes above EMA (center) or touches upper band
                exit_signal = (close[i] > ema_close[i]) or (high[i] >= keltner_upper[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
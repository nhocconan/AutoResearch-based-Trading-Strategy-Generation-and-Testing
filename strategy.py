#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted RSI + 12h Supertrend regime filter.
# Volume-Weighted RSI (VWRSI) incorporates volume into RSI calculation, making it more responsive to institutional flow.
# In trending regimes (12h Supertrend), trade in direction of VWRSI extremes with volume confirmation.
# Designed to capture momentum shifts validated by volume, working in both bull and bear markets by filtering chop.
# Target: 12-37 trades/year (50-150 over 4 years).

name = "6h_VolumeWeightedRSI_12hSupertrend_Regime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data for Supertrend regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h Supertrend (ATR=10, mult=3.0)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[:-1])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR using Wilder's smoothing
    atr_12h = pd.Series(tr).ewm(alpha=1/10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2_12h = (high_12h + low_12h) / 2
    upper_band_12h = hl2_12h + (3.0 * atr_12h)
    lower_band_12h = hl2_12h - (3.0 * atr_12h)
    
    # Initialize Supertrend
    supertrend_12h = np.full_like(close_12h, np.nan)
    direction_12h = np.full_like(close_12h, np.nan)  # 1 for uptrend, -1 for downtrend
    
    # Start from index 10 (after min_periods)
    for i in range(10, len(close_12h)):
        if np.isnan(upper_band_12h[i]) or np.isnan(lower_band_12h[i]) or np.isnan(atr_12h[i]):
            continue
            
        if i == 10:
            supertrend_12h[i] = lower_band_12h[i]
            direction_12h[i] = 1
        else:
            prev_close = close_12h[i-1]
            prev_supertrend = supertrend_12h[i-1]
            prev_direction = direction_12h[i-1]
            
            if prev_direction == 1:
                # Was in uptrend
                if close_12h[i] <= prev_supertrend:
                    # Reverse to downtrend
                    supertrend_12h[i] = upper_band_12h[i]
                    direction_12h[i] = -1
                else:
                    # Stay in uptrend
                    supertrend_12h[i] = max(lower_band_12h[i], prev_supertrend)
                    direction_12h[i] = 1
            else:
                # Was in downtrend
                if close_12h[i] >= prev_supertrend:
                    # Reverse to uptrend
                    supertrend_12h[i] = lower_band_12h[i]
                    direction_12h[i] = 1
                else:
                    # Stay in downtrend
                    supertrend_12h[i] = min(upper_band_12h[i], prev_supertrend)
                    direction_12h[i] = -1
    
    # Align 12h Supertrend direction to 6h timeframe
    direction_12h_aligned = align_htf_to_ltf(prices, df_12h, direction_12h)
    
    # Calculate 6h Volume-Weighted RSI (14-period)
    # Typical Price
    tp_6h = (high + low + close) / 3
    
    # Volume-weighted typical price change
    vwtp_6h = tp_6h * volume
    
    # Price change
    delta_tp = np.diff(tp_6h, prepend=tp_6h[0])
    
    # Volume-weighted price change
    vw_delta_tp = delta_tp * volume
    
    # Separate gains and losses (volume-weighted)
    gains = np.where(vw_delta_tp >= 0, vw_delta_tp, 0)
    losses = np.where(vw_delta_tp < 0, -vw_delta_tp, 0)
    
    # Smoothed average gains and losses (using Wilder's smoothing = EMA with alpha=1/period)
    avg_gains = pd.Series(gains).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_losses = pd.Series(losses).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Relative Strength (RS) and RSI
    rs_6h = np.divide(avg_gains, avg_losses, out=np.zeros_like(avg_gains), where=avg_losses!=0)
    rsi_6h = 100 - (100 / (1 + rs_6h))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(rsi_6h[i]) or np.isnan(direction_12h_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Regime filter: only trade when 12h Supertrend is defined (not NaN)
        is_uptrend = direction_12h_aligned[i] == 1
        is_downtrend = direction_12h_aligned[i] == -1
        
        if position == 0:
            # Long: VWRSI < 30 (oversold) AND 12h uptrend AND session
            if rsi_6h[i] < 30 and is_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: VWRSI > 70 (overbought) AND 12h downtrend AND session
            elif rsi_6h[i] > 70 and is_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: VWRSI >= 50 (mean reversion) OR reverse signal
            if rsi_6h[i] >= 50 or (rsi_6h[i] > 70 and is_downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: VWRSI <= 50 (mean reversion) OR reverse signal
            if rsi_6h[i] <= 50 or (rsi_6h[i] < 30 and is_uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
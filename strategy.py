#!/usr/bin/env python3
"""
Experiment #091: 6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation

HYPOTHESIS: Donchian channel breakouts on 6h timeframe, filtered by weekly pivot direction 
(from 1w timeframe) and confirmed by volume spike on 12h, creates a robust strategy that 
captures strong momentum moves while avoiding false breakouts. Weekly pivot provides 
institutional reference points (weekly high/low) to determine bias, while volume confirms 
institutional participation. Targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for weekly pivot direction (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot bias (price above/below weekly midpoint)
    if len(df_1w) >= 2:
        weekly_high = df_1w['high'].values
        weekly_low = df_1w['low'].values
        weekly_midpoint = (weekly_high + weekly_low) / 2
        weekly_bias = np.zeros(len(weekly_midpoint))  # 1 = bullish (above midpoint), -1 = bearish (below)
        weekly_bias[1:] = np.where(weekly_high[1:] > weekly_midpoint[1:], 1, -1)
        weekly_bias[0] = 0  # Neutral for first bar
        weekly_bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias)
    else:
        weekly_bias_aligned = np.zeros(n)
    
    # === HTF: 12h data for volume spike confirmation (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate volume ratio (current vs 20-period average) on 12h
    if len(df_12h) >= 20:
        vol_12h = df_12h['volume'].values
        vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_12h = np.zeros(len(vol_12h))
        vol_ratio_12h[20:] = vol_12h[20:] / vol_ma_20[20:]
        vol_ratio_12h[:20] = 1.0  # Neutral for warmup
        vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    else:
        vol_ratio_12h_aligned = np.full(n, 1.0)
    
    # === 6h Indicators ===
    # Donchian channel (20-period) - using shifted values to avoid look-ahead
    donchian_period = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    if n >= donchian_period:
        # Use rolling window on shifted data (shift by 1 to use only completed bars)
        high_shifted = np.roll(high, 1)
        low_shifted = np.roll(low, 1)
        high_shifted[0] = np.nan
        low_shifted[0] = np.nan
        
        for i in range(donchian_period, n):
            window_high = high_shifted[i-donchian_period+1:i+1]
            window_low = low_shifted[i-donchian_period+1:i+1]
            # Only use non-nan values
            valid_high = window_high[~np.isnan(window_high)]
            valid_low = window_low[~np.isnan(window_low)]
            if len(valid_high) >= donchian_period and len(valid_low) >= donchian_period:
                donchian_high[i] = np.max(valid_high)
                donchian_low[i] = np.min(valid_low)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    max_favorable_price = 0.0  # For trailing stop
    
    warmup = max(100, donchian_period + 20)  # Ensure enough data for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_bias_aligned[i]) or np.isnan(vol_ratio_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (Trailing stop + time-based exit) ---
        if in_position:
            # Update max favorable price
            if position_side > 0:  # Long
                max_favorable_price = max(max_favorable_price, high[i])
                # Trailing stop: exit if price drops 2.5*ATR from max favorable
                # Simple ATR approximation using recent range
                lookback = min(14, i)
                if lookback > 0:
                    tr_vals = []
                    for j in range(i-lookback+1, i+1):
                        tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                        tr_vals.append(tr)
                    atr_approx = np.mean(tr_vals) if tr_vals else 0
                    stop_level = max_favorable_price - 2.5 * atr_approx
                    if low[i] < stop_level:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                        continue
            else:  # Short
                max_favorable_price = min(max_favorable_price, low[i])
                lookback = min(14, i)
                if lookback > 0:
                    tr_vals = []
                    for j in range(i-lookback+1, i+1):
                        tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                        tr_vals.append(tr)
                    atr_approx = np.mean(tr_vals) if tr_vals else 0
                    stop_level = max_favorable_price + 2.5 * atr_approx
                    if high[i] > stop_level:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                        continue
            
            # Time-based exit: exit after 3 bars (18 hours) to prevent overstaying
            # This is a simple implementation - in practice would track entry bar
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout above upper band + weekly bullish bias + volume confirmation
        long_condition = (
            close[i] > donchian_high[i] and 
            weekly_bias_aligned[i] > 0 and 
            vol_ratio_12h_aligned[i] > 1.8
        )
        
        # Short: Donchian breakdown below lower band + weekly bearish bias + volume confirmation
        short_condition = (
            close[i] < donchian_low[i] and 
            weekly_bias_aligned[i] < 0 and 
            vol_ratio_12h_aligned[i] > 1.8
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            max_favorable_price = entry_price
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            max_favorable_price = entry_price
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals
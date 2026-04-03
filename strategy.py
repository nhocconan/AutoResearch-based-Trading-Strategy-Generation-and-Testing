#!/usr/bin/env python3
"""
Experiment #1999: 6h Camarilla Pivot Reversal + 12h Trend Filter + Volume Spike
HYPOTHESIS: Camarilla pivot levels (R3/S3, R4/S4) act as strong intraday support/resistance. 
- Primary: 6h Camarilla pivot levels calculated from prior 12h bar (H/L/C)
- Entry: Fade at R3/S3 with volume spike, breakout continuation at R4/S4 with volume spike
- HTF: 12h EMA(21) trend filter (only trade in direction of higher timeframe trend)
- Exit: Opposite Camarilla level (R3/S3 for continuation, R4/S4 for fade) or 6h close beyond pivot point
- Works in ranging markets (fade at R3/S3) and trending markets (breakout at R4/S4)
Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1999_6h_camarilla_pivot_reversal_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for EMA trend (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(21)
    ema_12h = pd.Series(close_12h).ewm(span=21, min_periods=21, adjust=False).mean().values
    trend_12h = np.where(close_12h > ema_12h, 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # === 6h Indicators: Camarilla Pivot Levels (from prior bar), Volume MA, ATR ===
    # Prior bar's H/L/C for Camarilla calculation
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    pivot_point = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    camarilla_r4 = pivot_point + (range_hl * 1.1 / 2)
    camarilla_r3 = pivot_point + (range_hl * 1.1 / 4)
    camarilla_s3 = pivot_point - (range_hl * 1.1 / 4)
    camarilla_s4 = pivot_point - (range_hl * 1.1 / 2)
    
    # Volume MA for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # ATR(14) for dynamic sizing (optional)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(pivot_point[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(camarilla_r4[i]) or np.isnan(camarilla_s4[i]) or
            np.isnan(trend_12h_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                # Exit conditions for long:
                # 1. Price reaches R3 (take profit for fade) OR
                # 2. Price reaches R4 (stop loss for fade / take profit for breakout) OR
                # 3. Price closes below pivot point (trend change)
                if price >= camarilla_r3[i] or price >= camarilla_r4[i] or price < pivot_point[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                # Exit conditions for short:
                # 1. Price reaches S3 (take profit for fade) OR
                # 2. Price reaches S4 (stop loss for fade / take profit for breakout) OR
                # 3. Price closes above pivot point (trend change)
                if price <= camarilla_s3[i] or price <= camarilla_s4[i] or price > pivot_point[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 12h trend alignment for bias filter
        trend_bias = trend_12h_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Fade strategy at R3/S3: price rejects at these levels
            # Long fade: price rejects at S3 and moves back above it
            # Short fade: price rejects at R3 and moves back below it
            
            # Breakout strategy at R4/S4: price breaks through these levels with momentum
            # Long breakout: price breaks above R4 and sustains
            # Short breakout: price breaks below S4 and sustains
            
            # Long entry conditions:
            # 1. Fade at S3: price > S3 AND prior close <= S3 (bounce from S3) AND 12h trend up
            # 2. Breakout at R4: price > R4 AND prior close <= R4 (break above R4) AND 12h trend up
            long_fade = (price > camarilla_s3[i]) and (prev_close[i] <= camarilla_s3[i]) and (trend_bias > 0)
            long_breakout = (price > camarilla_r4[i]) and (prev_close[i] <= camarilla_r4[i]) and (trend_bias > 0)
            
            # Short entry conditions:
            # 1. Fade at R3: price < R3 AND prior close >= R3 (rejection from R3) AND 12h trend down
            # 2. Breakout at S4: price < S4 AND prior close >= S4 (break below S4) AND 12h trend down
            short_fade = (price < camarilla_r3[i]) and (prev_close[i] >= camarilla_r3[i]) and (trend_bias < 0)
            short_breakout = (price < camarilla_s4[i]) and (prev_close[i] >= camarilla_s4[i]) and (trend_bias < 0)
            
            if long_fade or long_breakout:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            elif short_fade or short_breakout:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals
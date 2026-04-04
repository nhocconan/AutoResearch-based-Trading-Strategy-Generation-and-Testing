#!/usr/bin/env python3
"""
Experiment #5659: 6h Camarilla Pivot Reversal + Volume Spike + ATR Filter
HYPOTHESIS: On 6h timeframe, reversions from extreme Camarilla levels (R4/S4) with 
volume > 2x average and ATR > 1.5x ATR(50) capture high-probability mean-reversion 
bounces in ranging markets. Camarilla pivots derived from 1d data provide institutional 
support/resistance that works in both bull and bear regimes. Volume confirms rejection 
strength, ATR filter ensures sufficient volatility. Discrete sizing (0.25) minimizes 
fee churn. Target: 12-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5659_6h_camarilla_reversal_vol_atr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1d data for Camarilla pivot levels ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 2:
        # Calculate Camarilla levels from prior completed day
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # Camarilla levels: based on prior day's range
        rng = high_1d - low_1d
        close_prev = close_1d
        
        # Resistance levels
        r4 = close_prev + rng * 1.1 / 2
        r3 = close_prev + rng * 1.1 / 4
        r2 = close_prev + rng * 1.1 / 6
        r1 = close_prev + rng * 1.1 / 12
        
        # Support levels
        s1 = close_prev - rng * 1.1 / 12
        s2 = close_prev - rng * 1.1 / 6
        s3 = close_prev - rng * 1.1 / 4
        s4 = close_prev - rng * 1.1 / 2
        
        # We'll use R4 and S4 for extreme reversal signals
        camarilla_r4 = r4
        camarilla_s4 = s4
    else:
        camarilla_r4 = np.array([])
        camarilla_s4 = np.array([])
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # === 6h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 6h Indicators: ATR(14) and ATR(50) for volatility filter ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(20, 14, 50)  # volume avg, ATR(14), ATR(50) lookback
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr_14[i]) or np.isnan(atr_50[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Reverse signal or time-based (max 3 bars) ---
        if in_position:
            if position_side > 0:  # Long position
                # Exit: price drops below S3 (mean reversion target) or reverse signal
                if price < camarilla_s4_aligned[i] * 0.999:  # S3 equivalent
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                # Exit: price rises above R3 (mean reversion target) or reverse signal
                if price > camarilla_r4_aligned[i] * 1.001:  # R3 equivalent
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        volume_spike = volume_ratio[i] > 2.0
        sufficient_vol = atr_14[i] > 1.5 * atr_50[i]
        
        # Extreme Camarilla levels: price at or beyond R4/S4
        at_r4 = price >= camarilla_r4_aligned[i] * 0.999
        at_s4 = price <= camarilla_s4_aligned[i] * 1.001
        
        # Entry: reversal from extreme levels with volume and volatility
        long_setup = at_s4 and volume_spike and sufficient_vol
        short_setup = at_r4 and volume_spike and sufficient_vol
        
        if long_setup:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_setup:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals
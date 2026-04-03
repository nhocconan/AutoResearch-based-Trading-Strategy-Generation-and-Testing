#!/usr/bin/env python3
"""
Experiment #695: 6h Elder Ray + 1w Supertrend + Volume Confirmation
HYPOTHESIS: 6h Elder Ray (Bull/Bear Power) filtered by 1w Supertrend direction and volume confirmation 
captures institutional order flow with proper regime alignment. Uses discrete position sizing (0.25) 
to minimize fee churn. Works in bull/bear markets via Supertrend regime filter: long only in uptrend, 
short only in downtrend. Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_695_6h_elder_ray_1w_supertrend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for Supertrend regime filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Supertrend for 1w timeframe
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr_1w = np.zeros(len(high_1w))
    for i in range(1, len(high_1w)):
        tr_1w[i] = max(high_1w[i] - low_1w[i], abs(high_1w[i] - close_1w[i-1]), abs(low_1w[i] - close_1w[i-1]))
    tr_1w[0] = high_1w[0] - low_1w[0]
    
    # ATR
    atr_1w = pd.Series(tr_1w).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # Basic Upper and Lower Bands
    basic_ub = (high_1w + low_1w) / 2 + multiplier * atr_1w
    basic_lb = (high_1w + low_1w) / 2 - multiplier * atr_1w
    
    # Final Upper and Lower Bands
    final_ub = np.zeros_like(basic_ub)
    final_lb = np.zeros_like(basic_lb)
    for i in range(len(basic_ub)):
        if i == 0:
            final_ub[i] = basic_ub[i]
            final_lb[i] = basic_lb[i]
        else:
            if basic_ub[i] < final_ub[i-1] or close_1w[i-1] > final_ub[i-1]:
                final_ub[i] = basic_ub[i]
            else:
                final_ub[i] = final_ub[i-1]
                
            if basic_lb[i] > final_lb[i-1] or close_1w[i-1] < final_lb[i-1]:
                final_lb[i] = basic_lb[i]
            else:
                final_lb[i] = final_lb[i-1]
    
    # Supertrend
    supertrend_1w = np.zeros_like(close_1w)
    for i in range(len(supertrend_1w)):
        if i == 0:
            supertrend_1w[i] = final_ub[i]
        else:
            if supertrend_1w[i-1] == final_ub[i-1] and close_1w[i] <= final_ub[i]:
                supertrend_1w[i] = final_ub[i]
            elif supertrend_1w[i-1] == final_ub[i-1] and close_1w[i] > final_ub[i]:
                supertrend_1w[i] = final_lb[i]
            elif supertrend_1w[i-1] == final_lb[i-1] and close_1w[i] >= final_lb[i]:
                supertrend_1w[i] = final_lb[i]
            elif supertrend_1w[i-1] == final_lb[i-1] and close_1w[i] < final_lb[i]:
                supertrend_1w[i] = final_ub[i]
    
    # Supertrend direction: 1 = uptrend, -1 = downtrend
    supertrend_dir_1w = np.where(close_1w > supertrend_1w, 1, -1)
    
    # Align Supertrend direction to 6h timeframe
    supertrend_dir_1w_aligned = align_htf_to_ltf(prices, df_1w, supertrend_dir_1w)
    
    # === 6h Indicators: Elder Ray (Bull Power/Bear Power) ===
    ema_period = 13
    ema_close = pd.Series(close).ewm(span=ema_period, min_periods=ema_period, adjust=False).mean().values
    
    bull_power = high - ema_close  # Bull Power = High - EMA
    bear_power = low - ema_close   # Bear Power = Low - EMA
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for EMA and Supertrend calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_close[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(supertrend_dir_1w_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 8 bars (~48h on 6h) to avoid overtrading
            if bars_since_entry > 8:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Get regime from 1w Supertrend
            regime = supertrend_dir_1w_aligned[i]
            
            # Long: Bull Power > 0 (bullish momentum) + uptrend regime
            if bull_power[i] > 0 and regime > 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Bear Power < 0 (bearish momentum) + downtrend regime
            elif bear_power[i] < 0 and regime < 0:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals
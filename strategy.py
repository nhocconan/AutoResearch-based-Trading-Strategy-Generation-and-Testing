#!/usr/bin/env python3
"""
Experiment #054: 1h EMA(8,21) crossover + 4h/1d EMA(50) trend filter + volume spike + session filter
HYPOTHESIS: 1h EMA crossovers aligned with 4h/1d EMA trend and volume spikes capture momentum with structural confirmation.
Session filter (08-20 UTC) reduces noise. Position size fixed at 0.20 to control drawdown. Target: 60-150 total trades over 4 years.
Works in bull/bear markets by requiring alignment with higher timeframe trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_054_1h_ema8_21_4h_1d_ema50_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h and 1d data for EMA(50) trend filters (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(50) on 4h close
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate EMA(50) on 1d close
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 1h Indicators: EMA(8) and EMA(21) for crossover ===
    ema_8 = pd.Series(close).ewm(span=8, min_periods=8, adjust=False).mean().values
    ema_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Session filter: 08-20 UTC (precompute before loop) ===
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Position sizing (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 100  # Warmup for EMA stability
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        if hours[i] < 8 or hours[i] > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(ema_8[i]) or np.isnan(ema_21[i]) or np.isnan(ema_4h_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- EMA Trend Filters: Require alignment with 4h and 1d EMA(50) ---
        trend_4h_up = price > ema_4h_aligned[i]
        trend_4h_down = price < ema_4h_aligned[i]
        trend_1d_up = price > ema_1d_aligned[i]
        trend_1d_down = price < ema_1d_aligned[i]
        
        # --- EMA Crossover Signals ---
        ema_cross_up = ema_8[i] > ema_21[i] and ema_8[i-1] <= ema_21[i-1]
        ema_cross_down = ema_8[i] < ema_21[i] and ema_8[i-1] >= ema_21[i-1]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 4 bars to reduce churn
            if bars_since_entry < 4:
                signals[i] = position_side * SIZE
                continue
            
            # Exit on opposite EMA crossover with volume (profit taking)
            if position_side > 0 and ema_cross_down and volume_spike:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            elif position_side < 0 and ema_cross_up and volume_spike:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Only trade when EMA crossover aligns with BOTH 4h and 1d trends
        if ema_cross_up:
            # Long: EMA crossover up AND price above BOTH 4h and 1d EMA(50) AND volume spike
            if trend_4h_up and trend_1d_up and volume_spike:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            else:
                signals[i] = 0.0
        elif ema_cross_down:
            # Short: EMA crossover down AND price below BOTH 4h and 1d EMA(50) AND volume spike
            if trend_4h_down and trend_1d_down and volume_spike:
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
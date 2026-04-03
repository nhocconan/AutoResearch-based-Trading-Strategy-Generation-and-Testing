#!/usr/bin/env python3
"""
Experiment #067: 6h Elder Ray + 1d/1w regime filter + volume confirmation
HYPOTHESIS: Elder Ray (Bull/Bear Power) measures buying/selling pressure relative to EMA13.
In bull markets (1w EMA50 > 1d EMA200), go long when Bull Power > 0 and volume > 1.5x MA20.
In bear markets (1w EMA50 < 1d EMA200), go short when Bear Power < 0 and volume > 1.5x MA20.
Uses 6h timeframe for entries with 1d/1w HTF regime filter. Targets 75-150 total trades over 4 years
(19-37/year) with discrete sizing (0.25) and ATR(14) stoploss (2.5*ATR). Works in both bull/bear
by adapting to regime: trend-following in strong regimes, avoids whipsaws in chop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_067_6h_elder_ray_1d_1w_regime_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d and 1w data for regime filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate EMAs for regime detection
    ema_1d_200 = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_1w_50 = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_200_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_200)
    ema_1w_50_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_50)
    
    # === 6h Indicators: EMA13 for Elder Ray ===
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # === 6h Indicators: Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) ===
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)  # default to 1.0 for warmup period
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 200  # sufficient for 200-period EMA warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(ema_1d_200_aligned[i]) or
            np.isnan(ema_1w_50_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Regime Filter: 1w EMA50 vs 1d EMA200 ---
        bull_regime = ema_1w_50_aligned[i] > ema_1d_200_aligned[i]  # Bull market
        bear_regime = ema_1w_50_aligned[i] < ema_1d_200_aligned[i]  # Bear market
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5*ATR below entry
                stop_level = entry_price - 2.5 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.5*ATR above entry
                stop_level = entry_price + 2.5 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 12 bars (~3 days on 6h) to avoid overtrading
            if bars_since_entry > 12:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Bull regime: Long when Bull Power > 0 (buying pressure)
            if bull_regime and bull_power[i] > 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Bear regime: Short when Bear Power < 0 (selling pressure)
            elif bear_regime and bear_power[i] < 0:
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
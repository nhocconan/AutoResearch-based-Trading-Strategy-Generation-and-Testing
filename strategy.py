#!/usr/bin/env python3
"""
Experiment #079: 6h ATR Breakout + 12h Volume Regime + 1d Trend Filter

HYPOTHESAT: 6h ATR breakouts (price > highest(high,20) + k*ATR) with 12h volume regime filter 
(high volume = institutional participation) and 1d trend filter (price > EMA200) captures 
strong momentum moves in both bull and bear markets. The volume regime distinguishes 
between random breakouts and institutional-driven moves, while the 1d EMA200 filter ensures 
alignment with the primary trend. Targets 12-37 trades/year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_atr_breakout_vol_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for volume regime (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate volume regime: ratio of current volume to 50-period average
    if len(df_12h) >= 50:
        vol_12h = df_12h['volume'].values
        vol_ma_50 = pd.Series(vol_12h).rolling(window=50, min_periods=50).mean().values
        vol_regime_12h = np.zeros(len(vol_12h))
        vol_regime_12h[50:] = vol_12h[50:] / vol_ma_50[50:]
        vol_regime_12h[:50] = 1.0  # Neutral for warmup
        vol_regime_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_regime_12h)
    else:
        vol_regime_12h_aligned = np.full(n, 1.0)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(200) on 1d close
    if len(df_1d) >= 200:
        close_1d = df_1d['close'].values
        ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
        ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    else:
        ema_200_1d_aligned = np.full(n, np.nan)
    
    # === 6h Indicators ===
    # Calculate ATR(14)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Calculate highest high and lowest low over 20 periods
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(vol_regime_12h_aligned[i]) or np.isnan(ema_200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in alignment with 1d EMA200 trend ---
        price_above_1d_ema = close[i] > ema_200_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_200_1d_aligned[i]
        
        # --- Volume Regime Filter: Require high volume regime (> 1.8x average) ---
        high_volume_regime = vol_regime_12h_aligned[i] > 1.8
        
        # --- Exit Logic (ATR-based trailing stop) ---
        if in_position:
            # Calculate ATR for stoploss (using current ATR)
            current_atr = atr_14[i]
            
            if position_side > 0:  # Long position
                # Trailing stop: highest high since entry - 2.5 * ATR
                # Simplified: use recent highest high
                recent_high = np.max(high[max(0, i-20):i+1]) if i >= 20 else high[i]
                stop_level = recent_high - 2.5 * current_atr
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Trailing stop: lowest low since entry + 2.5 * ATR
                recent_low = np.min(low[max(0, i-20):i+1]) if i >= 20 else low[i]
                stop_level = recent_low + 2.5 * current_atr
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above highest high + k*ATR with volume regime and uptrend
        long_breakout = close[i] > highest_high_20[i] + 0.5 * atr_14[i]
        long_condition = long_breakout and high_volume_regime and price_above_1d_ema
        
        # Short: Price breaks below lowest low - k*ATR with volume regime and downtrend
        short_breakout = close[i] < lowest_low_20[i] - 0.5 * atr_14[i]
        short_condition = short_breakout and high_volume_regime and price_below_1d_ema
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals
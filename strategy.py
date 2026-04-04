#!/usr/bin/env python3
"""
Experiment #4015: 6h Elder Ray + Weekly Regime + Volume Confirmation
HYPOTHESIS: Elder Ray (Bull/Bear Power) identifies institutional buying/selling pressure. Weekly trend filter (EMA50) ensures alignment with higher timeframe direction. Volume > 1.8x MA20 confirms participation. Discrete sizing (0.25) and ATR(14) trailing stop (2.5x) control risk. Target: 75-150 total trades over 4 years (19-37/year). Works in both bull (buy strength) and bear (sell weakness) regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4015_6h_elder_ray_1w_regime_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for weekly EMA50 trend filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 1:
        weekly_ema50 = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
        weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    else:
        weekly_ema50_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: EMA13 for Elder Ray (Bull/Bear Power) ===
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(13 + 5, 20 + 5, 14 + 5, 1 + 5)  # EMA13, vol MA, ATR, HTF buffer
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema13[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(weekly_ema50_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.8x average) to filter noise
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
            bull_power = high[i] - ema13[i]
            bear_power = low[i] - ema13[i]
            
            # Weekly trend filter: price above/below weekly EMA50
            weekly_uptrend = price > weekly_ema50_aligned[i]
            weekly_downtrend = price < weekly_ema50_aligned[i]
            
            # Long: Bull Power > 0 (buying pressure) AND weekly uptrend
            if bull_power > 0 and weekly_uptrend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short: Bear Power < 0 (selling pressure) AND weekly downtrend
            elif bear_power < 0 and weekly_downtrend:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals
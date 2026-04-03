#!/usr/bin/env python3
"""
Experiment #255: 6h Elder Ray + 1w Regime + Volume Spike Strategy

HYPOTHESIS: Elder Ray (Bull/Bear Power) on 6h captures institutional buying/selling pressure. 
Combined with 1w trend filter (price vs EMA50) and volume confirmation (>1.8x average), 
this strategy identifies strong momentum moves with follow-through. In bull regimes (price > weekly EMA50), 
we take long signals when Bull Power > 0 and rising. In bear regimes (price < weekly EMA50), 
we take short signals when Bear Power < 0 and falling. Uses ATR-based stoploss (2.0x) and 
minimum 3-bar holding period. Target: 80-140 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_255_6h_elder_ray_1w_regime_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for regime detection (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for regime filter
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === 6h Indicators: EMA13 for Elder Ray calculation ===
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray Components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Smooth the power signals (EMA8)
    bull_power_smooth = pd.Series(bull_power).ewm(span=8, min_periods=8, adjust=False).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=8, min_periods=8, adjust=False).mean().values
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr_6h = np.zeros(n)
    tr_6h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_6h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr_6h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 50  # Warmup for 1w EMA50 stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(bull_power_smooth[i]) or
            np.isnan(bear_power_smooth[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- 1w Regime Filter: Bull regime if price > weekly EMA50, Bear regime if price < weekly EMA50 ---
        price = close[i]
        is_bull_regime = price > ema50_1w_aligned[i]
        is_bear_regime = price < ema50_1w_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Elder Ray Signals with Momentum ---
        # Bull signal: Bull Power > 0 AND rising (current > previous)
        bull_signal = bull_power_smooth[i] > 0 and bull_power_smooth[i] > bull_power_smooth[i-1]
        # Bear signal: Bear Power < 0 AND falling (current < previous)
        bear_signal = bear_power_smooth[i] < 0 and bear_power_smooth[i] < bear_power_smooth[i-1]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on opposite Elder Ray signal
                if bear_signal and volume_spike:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on opposite Elder Ray signal
                if bull_signal and volume_spike:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Bull regime: look for long signals
        if is_bull_regime:
            if bull_signal and volume_spike:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            else:
                signals[i] = 0.0
        # Bear regime: look for short signals
        elif is_bear_regime:
            if bear_signal and volume_spike:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        # In transition zones (price == EMA50), stay flat
        else:
            signals[i] = 0.0
    
    return signals
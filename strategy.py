#!/usr/bin/env python3
"""
Experiment #2259: 6h Elder Ray + ATR Regime Filter + Volume Confirmation
HYPOTHESIS: Elder Ray (Bull/Bear Power) combined with ATR-based regime detection captures
institutional buying/selling pressure while avoiding choppy markets. Works in both bull
(trend continuation) and bear (mean reversion at extremes) markets by adapting to volatility
regime. Uses 6h primary with 12h HTF for trend context and 1d for volume average.
Target: 75-150 total trades over 4 years (19-37/year) - optimized for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2259_6h_elder_ray_atr_regime_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for EMA trend context (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(50) for trend filter
    ema_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_12h = np.where(close_12h > ema_12h, 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # === HTF: 1d data for volume average (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # === 6h Indicators: Elder Ray, ATR, EMA(20) for dynamic levels ===
    # EMA(20) for Elder Ray calculation
    ema_20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Elder Ray Components
    bull_power = high - ema_20      # Bull Power = High - EMA(20)
    bear_power = low - ema_20       # Bear Power = Low - EMA(20)
    
    # ATR(14) for volatility regime and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR Regime: ATR(14) / ATR(50) ratio to detect volatility expansion/contraction
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = np.ones(n)
    valid_atr = (atr_50 > 0) & ~np.isnan(atr_50)
    atr_ratio[valid_atr] = atr[valid_atr] / atr_50[valid_atr]
    
    # Volume confirmation: current 6h volume vs 1d average
    vol_ratio = np.ones(n)
    valid_vol = (vol_ma_1d_aligned > 0) & ~np.isnan(vol_ma_1d_aligned)
    vol_ratio[valid_vol] = volume[valid_vol] / vol_ma_1d_aligned[valid_vol]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 100  # sufficient for all indicators (ATR50, EMA20)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_20[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(atr[i]) or np.isnan(atr_ratio[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(trend_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (wider stop for 6h)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if Bear Power turns strongly negative (selling pressure)
                elif bear_power[i] < -1.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if Bull Power turns strongly positive (buying pressure)
                elif bull_power[i] > 1.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Regime filter: only trade in expanding volatility (ATR ratio > 0.8)
        vol_expanding = atr_ratio[i] > 0.8
        
        # Volume confirmation: require volume > 1.2x 1d average
        volume_ok = vol_ratio[i] > 1.2
        
        if vol_expanding and volume_ok:
            # Long entry: Bull Power rising AND 12h trend up
            # Bull Power rising = current > previous
            bull_rising = bull_power[i] > bull_power[i-1]
            if bull_rising and trend_12h_aligned[i] > 0 and bull_power[i] > 0.5 * atr[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: Bear Power falling AND 12h trend down
            # Bear Power falling = current < previous (more negative)
            bear_falling = bear_power[i] < bear_power[i-1]
            elif bear_falling and trend_12h_aligned[i] < 0 and bear_power[i] < -0.5 * atr[i]:
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
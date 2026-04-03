#!/usr/bin/env python3
"""
Experiment #227: 6h Alligator + Elder Ray + 1d Trend Filter
HYPOTHESIS: Combines Bill Williams Alligator (trend detection) with Elder Ray (bull/bear power) on 6h timeframe, filtered by 1d EMA200 trend direction. Alligator jaws-teeth-lips alignment confirms trend strength, while Elder Ray measures buying/selling pressure. In bull markets (price > 1d EMA200), we go long when Elder Bull Power > 0 and Alligator is aligned (jaws < teeth < lips). In bear markets (price < 1d EMA200), we go short when Elder Bear Power < 0 and Alligator aligned inversely (jaws > teeth > lips). Volume confirmation (>1.5x average) filters weak signals. Discrete position sizing (0.25) balances return and fee drag. Target: 75-150 total trades over 4 years (19-37/year). Works in bull markets via trend-following entries and in bear markets via inverse logic, with symmetry for longs/shorts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_227_6h_alligator_elder_ray_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA200 trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA200
    ema200_1d = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # === 6h Indicators: Alligator (Jaws=13, Teeth=8, Lips=5) ===
    # Alligator uses SMMA (Smoothed Moving Average) which is EMA with alpha=1/period
    def smma(values, period):
        return pd.Series(values).ewm(alpha=1.0/period, min_periods=period, adjust=False).mean().values
    
    jaws = smma(high + low, 13)  # Typically uses median price (H+L)/2
    teeth = smma(high + low, 8)
    lips = smma(high + low, 5)
    
    # === 6h Indicators: Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) ===
    ema13 = smma(close, 13)
    bull_power = high - ema13
    bear_power = low - ema13
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 200  # Need enough data for EMA200 on 1d
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(ema200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Alligator Alignment Check ---
        # Bullish alignment: jaws < teeth < lips (alligator eating up)
        # Bearish alignment: jaws > teeth > lips (alligator eating down)
        bullish_aligned = jaws[i] < teeth[i] and teeth[i] < lips[i]
        bearish_aligned = jaws[i] > teeth[i] and teeth[i] > lips[i]
        
        # --- Elder Ray Power Check ---
        strong_bull = bull_power[i] > 0
        strong_bear = bear_power[i] < 0
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- 1d Trend Filter ---
        uptrend_1d = price > ema200_1d_aligned[i]
        downtrend_1d = price < ema200_1d_aligned[i]
        
        # --- Exit Logic (ATR-based stoploss using Alligator width as proxy) ---
        if in_position:
            bars_since_entry += 1
            
            # Calculate dynamic stop based on Alligator mouth width
            alligator_width = abs(lips[i] - jaws[i])
            stop_distance = max(alligator_width * 2.0, (high[i] - low[i]) * 1.5)  # At least 1.5x current bar range
            
            if position_side > 0:  # Long position
                stop_level = entry_price - stop_distance
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit if alignment breaks or Elder Ray turns negative
                if not (bullish_aligned and strong_bull):
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + stop_distance
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit if alignment breaks or Elder Ray turns positive
                if not (bearish_aligned and strong_bear):
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Require volume spike + Alligator alignment + Elder Ray confirmation
        if volume_spike:
            # Long: 1d uptrend + Alligator bullish aligned + Elder Bull Power > 0
            if uptrend_1d and bullish_aligned and strong_bull:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: 1d downtrend + Alligator bearish aligned + Elder Bear Power < 0
            elif downtrend_1d and bearish_aligned and strong_bear:
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
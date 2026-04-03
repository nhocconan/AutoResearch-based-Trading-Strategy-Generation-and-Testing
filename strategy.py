#!/usr/bin/env python3
"""
Experiment #247: 6h Elder Ray + 1d Alligator Trend (Bull/Bear Power + Jaw/Teeth/Lips)
HYPOTHESIS: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures buying/selling pressure. 
Alligator (Jaw=EMA13, Teeth=EMA8, Lips=EMA5) on 1d defines regime: 
- Bull: Lips > Teeth > Jaw (trending up) → take Elder Ray longs when Bull Power > 0 and rising
- Bear: Jaw > Teeth > Lips (trending down) → take Elder Ray shorts when Bear Power > 0 and rising
- Range: Otherwise → fade extremes (Bull Power < -0.5*ATR for short, Bear Power < -0.5*ATR for long)
Volume confirmation (>1.5x average) filters weak signals. ATR stoploss (2.5x) manages risk. 
Discrete position sizing (0.25) balances return and fee drag. Target: 100-200 total trades over 4 years (25-50/year).
Works in bull markets via trend-following with Elder Ray and in bear markets via mean reversion at extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_247_6h_elder_ray_1d_alligator_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Alligator (EMA 5,8,13) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_5 = pd.Series(close_1d).ewm(span=5, min_periods=5, adjust=False).mean().values
    ema_8 = pd.Series(close_1d).ewm(span=8, min_periods=8, adjust=False).mean().values
    ema_13 = pd.Series(close_1d).ewm(span=13, min_periods=13, adjust=False).mean().values
    lips_1d = align_htf_to_ltf(prices, df_1d, ema_5)
    teeth_1d = align_htf_to_ltf(prices, df_1d, ema_8)
    jaw_1d = align_htf_to_ltf(prices, df_1d, ema_13)
    
    # === 6h Indicators: EMA13 for Elder Ray ===
    ema_13_6h = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # === 6h Indicators: Elder Ray Components ===
    bull_power = high - ema_13_6h  # Buying pressure
    bear_power = ema_13_6h - low   # Selling pressure
    
    # === 6h Indicators: ATR(14) for stoploss and thresholds ===
    tr_6h = np.zeros(n)
    tr_6h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_6h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr_6h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
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
    
    warmup = 60
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(lips_1d[i]) or np.isnan(teeth_1d[i]) or np.isnan(jaw_1d[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Alligator Regime Detection (1d) ---
        # Bullish alignment: Lips > Teeth > Jaw
        bullish_alignment = (lips_1d[i] > teeth_1d[i]) and (teeth_1d[i] > jaw_1d[i])
        # Bearish alignment: Jaw > Teeth > Lips
        bearish_alignment = (jaw_1d[i] > teeth_1d[i]) and (teeth_1d[i] > lips_1d[i])
        # Range: otherwise
        
        # --- Elder Ray Signals with Thresholds ---
        # Bull Power rising: current > previous
        bull_power_rising = bull_power[i] > bull_power[i-1]
        # Bear Power rising: current > previous
        bear_power_rising = bear_power[i] > bear_power[i-1]
        # Extreme thresholds for mean reversion in ranging markets
        bull_extreme = bull_power[i] < -0.5 * atr_14[i]
        bear_extreme = bear_power[i] < -0.5 * atr_14[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit conditions: reversal signals
                if bearish_alignment and bear_power_rising and volume_spike:
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
                # Exit conditions: reversal signals
                if bullish_alignment and bull_power_rising and volume_spike:
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
        if volume_spike:
            # Regime-based entries
            if bullish_alignment:
                # Bull trend: go long on rising Bull Power
                if bull_power_rising and bull_power[i] > 0:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
            elif bearish_alignment:
                # Bear trend: go short on rising Bear Power
                if bear_power_rising and bear_power[i] > 0:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
            else:
                # Range market: mean reversion at extremes
                if bull_extreme:
                    # Extreme selling pressure -> long reversion
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                elif bear_extreme:
                    # Extreme buying pressure -> short reversion
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
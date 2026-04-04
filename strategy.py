#!/usr/bin/env python3
"""
Experiment #3171: 6h Williams Alligator + Elder Ray Power + 1d Regime Filter
HYPOTHESIS: Combines Williams Alligator (trend detection) with Elder Ray (bull/bear power) on 6h timeframe,
filtered by 1d market regime (trending vs ranging) using ADX. Alligator identifies trend direction and strength,
Elder Ray measures buying/selling pressure, and 1d ADX regime filter ensures trades only occur in favorable market conditions.
Designed to work in both bull (trend following) and bear (mean reversion from extremes) markets by adapting to regime.
Target: 75-150 total trades over 4 years (19-37/year) with discrete position sizing to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3171_6h_alligator_elder_ray_1d_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for ADX regime filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d data
    def calculate_adx(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(high, np.nan)
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
        # Smoothed values
        tr_period = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
        dm_plus_period = pd.Series(dm_plus).rolling(window=period, min_periods=period).mean().values
        dm_minus_period = pd.Series(dm_minus).rolling(window=period, min_periods=period).mean().values
        # Directional Indicators
        di_plus = 100 * dm_plus_period / tr_period
        di_minus = 100 * dm_minus_period / tr_period
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 6h Indicators: Williams Alligator (Jaw, Teeth, Lips) ===
    def calculate_alligator(high, low, close, jaw_period=13, teeth_period=8, lips_period=5):
        if len(close) < jaw_period:
            return (np.full_like(close, np.nan), np.full_like(close, np.nan), np.full_like(close, np.nan))
        # Median price
        median_price = (high + low) / 2
        # Jaw (Blue) - 13-period SMMA of median, shifted 8 bars
        jaw = pd.Series(median_price).rolling(window=jaw_period, min_periods=jaw_period).mean().values
        jaw = np.concatenate([np.full(8, np.nan), jaw[:-8]]) if len(jaw) > 8 else np.full_like(jaw, np.nan)
        # Teeth (Red) - 8-period SMMA of median, shifted 5 bars
        teeth = pd.Series(median_price).rolling(window=teeth_period, min_periods=teeth_period).mean().values
        teeth = np.concatenate([np.full(5, np.nan), teeth[:-5]]) if len(teeth) > 5 else np.full_like(teeth, np.nan)
        # Lips (Green) - 5-period SMMA of median, shifted 3 bars
        lips = pd.Series(median_price).rolling(window=lips_period, min_periods=lips_period).mean().values
        lips = np.concatenate([np.full(3, np.nan), lips[:-3]]) if len(lips) > 3 else np.full_like(lips, np.nan)
        return jaw, teeth, lips
    
    jaw, teeth, lips = calculate_alligator(high, low, close)
    
    # === 6h Indicators: Elder Ray Power (Bull Power, Bear Power) ===
    def calculate_elder_ray(high, low, close, ema_period=13):
        if len(close) < ema_period:
            return (np.full_like(close, np.nan), np.full_like(close, np.nan))
        ema_close = pd.Series(close).ewm(span=ema_period, adjust=False).mean().values
        bull_power = high - ema_close
        bear_power = low - ema_close
        return bull_power, bear_power
    
    bull_power, bear_power = calculate_elder_ray(high, low, close, 13)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    
    warmup = max(50, 13, 8, 5, 13)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in trending markets (ADX > 25) ---
        if adx_1d_aligned[i] < 25:
            signals[i] = 0.0
            continue
        
        # --- Alligator Trend Detection ---
        # Alligator is aligned (trending) when: Lips > Teeth > Jaw (bullish) or Lips < Teeth < Jaw (bearish)
        bullish_aligned = lips[i] > teeth[i] > jaw[i]
        bearish_aligned = lips[i] < teeth[i] < jaw[i]
        
        # --- Elder Ray Power Signals ---
        # Bull Power > 0 indicates buying pressure, Bear Power < 0 indicates selling pressure
        strong_bull_power = bull_power[i] > 0
        strong_bear_power = bear_power[i] < 0
        
        # --- Entry Logic ---
        if not in_position:
            # Long entry: Alligator bullish aligned + Bull Power positive
            if bullish_aligned and strong_bull_power:
                in_position = True
                position_side = 1
                signals[i] = SIZE
            # Short entry: Alligator bearish aligned + Bear Power negative
            elif bearish_aligned and strong_bear_power:
                in_position = True
                position_side = -1
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            # --- Exit Logic ---
            # Exit long position: Alligator loses bullish alignment OR Bull Power turns negative
            if position_side == 1:
                if not (bullish_aligned and strong_bull_power):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            # Exit short position: Alligator loses bearish alignment OR Bear Power turns positive
            else:  # position_side == -1
                if not (bearish_aligned and strong_bear_power):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
    
    return signals
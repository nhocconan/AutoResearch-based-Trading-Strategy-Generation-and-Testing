#!/usr/bin/env python3
"""
Experiment #4375: 6h Williams Alligator + Elder Ray + 1d Regime Filter
HYPOTHESIS: Williams Alligator (jaw/teeth/lips) identifies trend phase on 6h, Elder Ray (bull/bear power) measures momentum strength, and 1d ADX regime filter ensures we only trade in strong trends (ADX>25) or mean-revert in ranges (ADX<20). This combination avoids whipsaws in sideways markets while capturing strong moves. Works in bull via buying dips in uptrends (teeth>jaw, bull power>0), in bear via selling rallies in downtrends (teeth<jaw, bear power<0). Position size 0.25 targets 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4375_6h_alligator_elder_ray_1d_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(open_time).hour
    
    # === Precompute HTF: 1d ADX for regime filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 30:
        # Calculate ADX(14) on 1d
        h_1d = df_1d['high'].values
        l_1d = df_1d['low'].values
        c_1d = df_1d['close'].values
        
        # True Range
        tr1 = h_1d[1:] - l_1d[1:]
        tr2 = np.abs(h_1d[1:] - c_1d[:-1])
        tr3 = np.abs(l_1d[1:] - c_1d[:-1])
        tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        up_move = h_1d[1:] - h_1d[:-1]
        down_move = l_1d[:-1] - l_1d[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed TR, +DM, -DM
        tr_ma = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
        plus_dm_ma = pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
        minus_dm_ma = pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Directional Indicators
        plus_di = 100 * plus_dm_ma / tr_ma
        minus_di = 100 * minus_dm_ma / tr_ma
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx_1d = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
        adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    else:
        adx_1d_aligned = np.full(n, 20.0)  # Default to range regime if insufficient data
    
    # === 6h Indicators: Williams Alligator (SMMA) ===
    # Jaw: SMMA(13, 8), Teeth: SMMA(8, 5), Lips: SMMA(5, 3)
    def smma(arr, period):
        """Smoothed Moving Average"""
        sma = pd.Series(arr).rolling(window=period, min_periods=period).mean().values
        smma = np.full_like(arr, np.nan, dtype=np.float64)
        smma[period-1] = sma[period-1]
        for i in range(period, len(arr)):
            if not np.isnan(smma[i-1]) and not np.isnan(sma[i]):
                smma[i] = (smma[i-1] * (period-1) + sma[i]) / period
            else:
                smma[i] = np.nan
        return smma
    
    jaw = smma((high + low) / 2, 13)  # SMMA of median price, period 13
    jaw = np.roll(jaw, 8)  # Shift 8 bars forward
    
    teeth = smma((high + low) / 2, 8)   # SMMA of median price, period 8
    teeth = np.roll(teeth, 5)   # Shift 5 bars forward
    
    lips = smma((high + low) / 2, 5)    # SMMA of median price, period 5
    lips = np.roll(lips, 3)   # Shift 3 bars forward
    
    # === 6h Indicators: Elder Ray (Bull/Bear Power) ===
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    warmup = max(30, 13+8, 13)  # ADX, Alligator, EMA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: 1d ADX ---
        adx = adx_1d_aligned[i]
        is_trending = adx > 25
        is_ranging = adx < 20
        
        # --- Alligator Signals ---
        # Alligator asleep: jaws, teeth, lips intertwined (no clear trend)
        alligator_asleep = (abs(jaw[i] - teeth[i]) < (teeth[i] - lips[i]) * 0.1) and \
                           (abs(teeth[i] - lips[i]) < (jaw[i] - teeth[i]) * 0.1)
        
        # Alligator awake and trending up: lips > teeth > jaw
        alligator_up = lips[i] > teeth[i] and teeth[i] > jaw[i]
        
        # Alligator awake and trending down: lips < teeth < jaw
        alligator_down = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # --- Elder Ray Signals ---
        strong_bull = bull_power[i] > 0 and bull_power[i] > bear_power[i] * 2
        strong_bear = bear_power[i] < 0 and abs(bear_power[i]) > bull_power[i] * 2
        
        # --- Entry Logic ---
        if not alligator_asleep:
            # In trending regime (ADX>25): follow Alligator + Elder Ray
            if is_trending:
                if alligator_up and strong_bull:
                    signals[i] = SIZE  # Long
                elif alligator_down and strong_bear:
                    signals[i] = -SIZE  # Short
                else:
                    signals[i] = 0.0
            # In ranging regime (ADX<20): fade extremes with Elder Ray
            elif is_ranging:
                if strong_bear and lips[i] < jaw[i]:  # Bear power but price below jaw (fade down)
                    signals[i] = SIZE  # Long (mean reversion up)
                elif strong_bull and lips[i] > jaw[i]:  # Bull power but price above jaw (fade up)
                    signals[i] = -SIZE  # Short (mean reversion down)
                else:
                    signals[i] = 0.0
            else:
                # Transition regime (ADX 20-25): reduce position or wait
                signals[i] = 0.0
        else:
            # Alligator asleep: no clear trend, stay out
            signals[i] = 0.0
    
    return signals
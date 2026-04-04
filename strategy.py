#!/usr/bin/env python3
"""
Experiment #4679: 6h Elder Ray + 12h ADX Regime Filter
HYPOTHESIS: Elder Ray (Bull/Bear Power) identifies momentum strength; 12h ADX > 25 filters for trending regimes.
Long when Bull Power > 0 and Bear Power < 0 (both positive momentum) in strong trend (ADX>25).
Short when Bear Power < 0 and Bull Power > 0 (both negative momentum) in strong trend.
Uses EMA(13) as trend reference. Works in bull (strong uptrend) and bear (strong downtrend).
Target: 12-37 trades/year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4679_6h_elder_ray_12h_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 12h data for ADX regime filter
    df_12h = get_htf_data(prices, '12h')
    
    # === 12h Indicators: ADX(14) for regime filter ===
    if len(df_12h) >= 14:
        # Calculate +DM, -DM, TR
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        close_12h = df_12h['close'].values
        
        up_move = np.diff(high_12h, prepend=high_12h[0])
        down_move = -np.diff(low_12h, prepend=low_12h[0])
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        tr1 = np.abs(np.diff(high_12h, prepend=high_12h[0]))
        tr2 = np.abs(np.diff(low_12h, prepend=low_12h[0]))
        tr3 = np.abs(np.diff(close_12h, prepend=close_12h[0]))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Wilder's smoothing (alpha = 1/period)
        def WilderSmoothing(data, period):
            alpha = 1.0 / period
            result = np.full_like(data, np.nan)
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            return result
        
        atr_12h = WilderSmoothing(tr, 14)
        plus_di_12h = 100 * WilderSmoothing(plus_dm, 14) / atr_12h
        minus_di_12h = 100 * WilderSmoothing(minus_dm, 14) / atr_12h
        dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h + 1e-10)
        adx_12h = WilderSmoothing(dx_12h, 14)
    else:
        adx_12h = np.full(len(df_12h), np.nan)
    
    # Align HTF ADX to 6h timeframe
    if len(adx_12h) > 0:
        adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    else:
        adx_12h_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: EMA(13) for trend reference ===
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # === 6h Indicators: Elder Ray (Bull Power, Bear Power) ===
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(13, 14)  # EMA, ADX warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in strong trends (ADX > 25) ---
        strong_trend = adx_12h_aligned[i] > 25
        
        # --- Elder Ray Signals ---
        # Bull Power > 0: ability to push prices above average (bullish momentum)
        # Bear Power < 0: ability to push prices below average (bearish momentum)
        bullish_momentum = bull_power[i] > 0
        bearish_momentum = bear_power[i] < 0
        
        # --- Entry Logic ---
        # Long: Both bullish AND bearish power show bullish bias in strong trend
        # (Bull Power > 0 AND Bear Power < 0) indicates underlying strength
        if strong_trend and bullish_momentum and bearish_momentum:
            if not in_position or position_side != 1:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            else:
                signals[i] = SIZE
        # Short: Both powers show bearish bias in strong trend
        elif strong_trend and (not bullish_momentum) and (not bearish_momentum):
            if not in_position or position_side != -1:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = -SIZE
        # --- Exit Logic: Flat when regime weakens or momentum diverges ---
        else:
            in_position = False
            position_side = 0
            signals[i] = 0.0
    
    return signals
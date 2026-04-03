#!/usr/bin/env python3
"""
Experiment #1535: 6h Williams Alligator + Elder Ray Power + Weekly Pivot Direction
HYPOTHESIS: Williams Alligator (JAW/TEETH/LIPS) defines trend regime, Elder Ray (Bull/Bear Power) measures momentum strength, and weekly pivot provides institutional bias. In 6h timeframe, this combination filters whipsaws in ranging markets while capturing strong trends. Weekly pivot direction ensures alignment with smart money flow. Position size 0.25 balances opportunity with drawdown control. Target: 80-180 total trades over 4 years (20-45/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1535_6h_alligator_elder_ray_wp_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours for filter (optional, can remove if too restrictive)
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 1w data for weekly pivot direction (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point calculation: (H + L + C) / 3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    # Weekly trend: price above/below pivot
    weekly_trend = np.where(close_1w > weekly_pivot, 1, -1)
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    # === 6h Indicators: Williams Alligator (Smoothed Medians) ===
    # JAW: 13-period SMMA, shifted 8 bars
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    # TEETH: 8-period SMMA, shifted 5 bars  
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    # LIPS: 5-period SMMA, shifted 3 bars
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Alligator alignment: JAW > TEETH > LIPS = uptrend, reverse = downtrend
    alligator_long = (jaw > teeth) & (teeth > lips)
    alligator_short = (jaw < teeth) & (teeth < lips)
    
    # === 6h Indicators: Elder Ray Power (Bull/Bear Power) ===
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Elder Ray signals: Bull Power > 0 AND rising = bullish momentum
    # Bear Power < 0 AND falling = bearish momentum
    bull_power_rising = np.zeros(n, dtype=bool)
    bear_power_falling = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        bull_power_rising[i] = bull_power[i] > bull_power[i-1]
        bear_power_falling[i] = bear_power[i] < bear_power[i-1]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 30  # sufficient for all indicators (JAW needs 13+8=21, plus shifting)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(weekly_trend_aligned[i]) or np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5*ATR below entry (wider for 6h volatility)
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
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require weekly pivot alignment with higher timeframe trend
        weekly_align = weekly_trend_aligned[i]
        
        # Alligator trend alignment
        alligator_align_long = alligator_long[i] and weekly_align > 0
        alligator_align_short = alligator_short[i] and weekly_align < 0
        
        # Elder Ray momentum confirmation
        elder_long = bull_power[i] > 0 and bull_power_rising[i]
        elder_short = bear_power[i] < 0 and bear_power_falling[i]
        
        # Enter long: Alligator uptrend + weekly bullish + Elder Ray bullish momentum
        if alligator_align_long and elder_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Enter short: Alligator downtrend + weekly bearish + Elder Ray bearish momentum
        elif alligator_align_short and elder_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals
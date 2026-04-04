#!/usr/bin/env python3
"""
Experiment #5267: 6h Williams Alligator + Elder Ray + 1d Regime Filter
HYPOTHESIS: On 6h timeframe, combining Williams Alligator (trend identification) with Elder Ray (bull/bear power) filtered by 1d trend direction captures strong momentum moves while avoiding whipsaws. The Alligator's jaw-teeth-lips alignment indicates trend strength, while Elder Ray measures buying/selling pressure. In bull regime (price > 1d EMA50), we go long when Elder Bull Power > 0 and Alligator is bullish (lips > teeth > jaw). In bear regime (price < 1d EMA50), we go short when Elder Bear Power < 0 and Alligator is bearish (jaw > teeth > lips). Uses discrete position sizing (0.25) to balance profit potential with drawdown control. Designed for 12-25 trades/year on 6h timeframe (50-100 total over 4 years) to minimize fee drag. Works in bull markets by catching strong uptrends and in bear markets by catching strong downtrends, while avoiding ranging conditions where Alligator is intertwined.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5267_6h_alligator_elder_ray_1d_regime_v1"
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
    
    # === HTF: 1d data for regime filter (EMA50) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 50:
        ema_50 = pd.Series(df_1d['close']).ewm(span=50, min_periods=50, adjust=False).mean().shift(1).values
        ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    else:
        ema_50_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Williams Alligator (13,8,5 SMAs) ===
    # Jaw: 13-period SMA, shifted 8 bars
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA, shifted 5 bars
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA, shifted 3 bars
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # === 6h Indicators: Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) ===
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema13  # Measures buying pressure
    bear_power = low - ema13   # Measures selling pressure (negative values indicate pressure)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(13, 8, 5, 50)  # Alligator, EMA13, EMA50 warmup
    
    for i in range(warmup, n):
        # --- Session Filter: 00-24 UTC (6h timeframe, less restrictive) ---
        # 6h candles already filter to specific sessions, so we can use full day
        # Optional: avoid low liquidity periods if needed
        
        # --- Data Validity Check ---
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Close position when Alligator reverses or regime changes ---
        if in_position:
            # Check for Alligator reversal (teeth crosses jaw)
            alligator_bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
            alligator_bearish = jaw[i] > teeth[i] and teeth[i] > lips[i]
            
            # Check regime consistency
            regime_bullish = price > ema_50_aligned[i]
            regime_bearish = price < ema_50_aligned[i]
            
            # Exit conditions:
            # 1. Alligator reverses direction
            # 2. Regime changes (price crosses 1d EMA50)
            # 3. Elder Power diverges (bull power < 0 in long, bear power > 0 in short)
            if position_side > 0:  # Long position
                if (not alligator_bullish) or (not regime_bullish) or (bull_power[i] < 0):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                if (not alligator_bearish) or (not regime_bearish) or (bear_power[i] > 0):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Alligator alignment
        alligator_bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_bearish = jaw[i] > teeth[i] and teeth[i] > lips[i]
        
        # Regime filter from 1d
        regime_bullish = price > ema_50_aligned[i]
        regime_bearish = price < ema_50_aligned[i]
        
        # Elder Ray confirmation
        elder_bullish = bull_power[i] > 0  # Buying pressure present
        elder_bearish = bear_power[i] < 0  # Selling pressure present
        
        # Entry conditions: Alligator aligned + regime match + Elder confirmation
        if alligator_bullish and regime_bullish and elder_bullish:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif alligator_bearish and regime_bearish and elder_bearish:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals
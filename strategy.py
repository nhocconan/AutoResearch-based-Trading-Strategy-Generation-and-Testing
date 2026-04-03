#!/usr/bin/env python3
"""
Experiment #2099: 6h Donchian(20) breakout + 12h pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts capture swing momentum. 
- Primary: 6h Donchian(20) breakout with volume > 1.8x 20-bar average
- HTF: 12h Camarilla pivot levels (R3/S3 for mean reversion fade, R4/S4 for breakout)
- Logic: In ranging markets (price between R3/S3), fade extremes. In trending markets (price breaks R4/S4), continuation.
- Works in bull/bear by adapting to 12h regime via pivot levels.
Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2099_6h_donchian20_12h_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for Camarilla pivots (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla pivots for each 12h bar
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R3 = Pivot + Range * 1.1/2
    # S3 = Pivot - Range * 1.1/2
    # R4 = Pivot + Range * 1.1
    # S4 = Pivot - Range * 1.1
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    r3_12h = pivot_12h + range_12h * 1.1 / 2.0
    s3_12h = pivot_12h - range_12h * 1.1 / 2.0
    r4_12h = pivot_12h + range_12h * 1.1
    s4_12h = pivot_12h - range_12h * 1.1
    
    # Regime detection: 
    # 0 = ranging (between S3 and R3)
    # 1 = bullish (above R4)
    # -1 = bearish (below S4)
    # 0.5 = weakening bullish (between R3 and R4)
    # -0.5 = weakening bearish (between S4 and S3)
    regime_12h = np.zeros_like(close_12h)
    regime_12h[close_12h > r4_12h] = 1
    regime_12h[close_12h < s4_12h] = -1
    regime_12h[(close_12h > r3_12h) & (close_12h <= r4_12h)] = 0.5
    regime_12h[(close_12h < s3_12h) & (close_12h >= s4_12h)] = -0.5
    
    # Align regimes to 6h timeframe
    regime_12h_aligned = align_htf_to_ltf(prices, df_12h, regime_12h)
    
    # === 6h Indicators: Donchian(20), Volume MA(20) ===
    # Donchian channels
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_ma
    donchian_lower = low_ma
    
    # Volume MA for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(regime_12h_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        regime = regime_12h_aligned[i]
        
        # --- Exit Logic (time-based or reversal) ---
        if in_position:
            # Exit conditions based on regime and price action
            if position_side > 0:  # Long position
                # Exit if price drops below Donchian lower (mean reversion in ranging)
                # or if regime turns bearish
                if price <= donchian_lower[i] or regime < -0.5:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                # Exit if price rises above Donchian upper (mean reversion in ranging)
                # or if regime turns bullish
                if price >= donchian_upper[i] or regime > 0.5:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Fading logic in ranging regime (regime between -0.5 and 0.5)
            if abs(regime) <= 0.5:
                # Fade at extremes: short near R3, long near S3
                if price >= donchian_upper[i] and regime >= 0:  # Near upper band in weak/strong bullish
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    signals[i] = -SIZE
                elif price <= donchian_lower[i] and regime <= 0:  # Near lower band in weak/strong bearish
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    signals[i] = SIZE
            # Continuation logic in trending regime
            elif regime > 0.5:  # Bullish regime
                if price > donchian_upper[i]:  # Breakout continuation
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    signals[i] = SIZE
            elif regime < -0.5:  # Bearish regime
                if price < donchian_lower[i]:  # Breakdown continuation
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals
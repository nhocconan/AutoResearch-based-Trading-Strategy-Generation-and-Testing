#!/usr/bin/env python3
"""
Experiment #5275: 6h Donchian Breakout + 1w Pivot Direction + Volume Confirmation
HYPOTHESIS: On 6h timeframe, price breaking above/below 20-period Donchian channel with confirmation from 1-week pivot direction (bullish if price > weekly pivot, bearish if price < weekly pivot) and volume spike (>1.5x 20-period average volume) captures strong momentum moves. Weekly pivot provides higher-timeframe bias to avoid counter-trend trades, while Donchian breakout captures breakouts from consolidation. Volume confirmation ensures breakouts have conviction. Designed for 12-30 trades/year on 6h timeframe (50-120 total over 4 years) to minimize fee drag. Works in bull markets by buying breakouts above weekly pivot and in bear markets by selling breakdowns below weekly pivot.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5275_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for pivot points (weekly high, low, close) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 1:
        # Calculate weekly pivot: P = (H + L + C) / 3
        weekly_pivot = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot.values)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    # Upper band: 20-period high
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume Spike (>1.5x 20-period average volume) ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(20, 20)  # Donchian, volume avg warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Close position when price reverses back into Donchian channel ---
        if in_position:
            if position_side > 0:  # Long position
                # Exit when price closes below Donchian lower band
                if price < donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                # Exit when price closes above Donchian upper band
                if price > donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Donchian breakout conditions
        breakout_up = price > donchian_high[i]  # Price breaks above upper band
        breakout_down = price < donchian_low[i]  # Price breaks below lower band
        
        # Weekly pivot direction filter
        pivot_bullish = price > weekly_pivot_aligned[i]  # Price above weekly pivot = bullish bias
        pivot_bearish = price < weekly_pivot_aligned[i]  # Price below weekly pivot = bearish bias
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry conditions: Donchian breakout + pivot direction match + volume confirmation
        if breakout_up and pivot_bullish and vol_confirm:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif breakout_down and pivot_bearish and vol_confirm:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals
#!/usr/bin/env python3
"""
Experiment #152: 12h Donchian Breakout + 1d Volume Spike + Weekly Pivot Direction

HYPOTHESIS: Donchian(20) breakouts on 12h timeframe with volume confirmation (>2x 20-period average volume on 1d) 
and weekly pivot direction filter (price above/below weekly pivot) captures strong momentum moves while avoiding 
counter-trend breakouts. Weekly pivot provides structural bias from higher timeframe (1d/1w), adapting to 
bull/bear markets. 12h timeframe targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag. 
Volume spike confirms institutional participation, reducing false breakouts. Works in both bull and bear via 
weekly pivot filter that ensures alignment with longer-term structure.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_breakout_1d_volume_weekly_pivot_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # === HTF: 1d data for volume spike filter and weekly pivot ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d volume spike: >2x 20-period average volume
    avg_vol_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (2.0 * avg_vol_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Calculate weekly pivot (using prior week's OHLC)
    # Approximate weekly by using 5 trading days (5x 1d bars)
    def calculate_weekly_pivot(high, low, close):
        # Need at least 5 days for weekly calculation
        weekly_high = np.full_like(high, np.nan)
        weekly_low = np.full_like(high, np.nan)
        weekly_close = np.full_like(high, np.nan)
        
        for i in range(len(close)):
            if i >= 4:  # Need 5 days: i-4 to i
                weekly_high[i] = np.max(high[i-4:i+1])
                weekly_low[i] = np.min(low[i-4:i+1])
                weekly_close[i] = close[i]  # Current day's close
        
        # Weekly pivot = (weekly_high + weekly_low + weekly_close) / 3
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        return weekly_pivot
    
    weekly_pivot_1d = calculate_weekly_pivot(high_1d, low_1d, close_1d)
    weekly_pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot_1d)
    
    # === 12h Indicators ===
    # Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(weekly_pivot_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Weekly Pivot Direction Filter ---
        price_vs_pivot = close[i] - weekly_pivot_1d_aligned[i]
        bullish_bias = price_vs_pivot > 0   # Price above weekly pivot
        bearish_bias = price_vs_pivot < 0   # Price below weekly pivot
        
        # --- Donchian Breakout + Volume Confirmation ---
        # Upper breakout: price breaks above Donchian high with volume spike
        upper_breakout = (close[i] > donchian_high[i]) and vol_spike_1d_aligned[i]
        # Lower breakout: price breaks below Donchian low with volume spike
        lower_breakout = (close[i] < donchian_low[i]) and vol_spike_1d_aligned[i]
        
        # --- Position Management (Exit Logic) ---
        if in_position:
            # Exit when price returns to Donchian midpoint (mean reversion within channel)
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2.0
            if position_side > 0:  # Long
                if close[i] < donchian_mid:
                    in_position = False
                    position_side = 0
            else:  # Short
                if close[i] > donchian_mid:
                    in_position = False
                    position_side = 0
            
            if not in_position:
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: upper breakout + bullish bias (price above weekly pivot)
        if upper_breakout and bullish_bias:
            in_position = True
            position_side = 1
            signals[i] = SIZE
        # Short: lower breakout + bearish bias (price below weekly pivot)
        elif lower_breakout and bearish_bias:
            in_position = True
            position_side = -1
            signals[i] = -SIZE
    
    return signals
</sub>
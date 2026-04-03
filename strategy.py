#!/usr/bin/env python3
"""
Experiment #139: 6h Donchian Breakout + 12h Volume Spike + Weekly Pivot Direction

HYPOTHESIS: Donchian(20) breakouts on 6h with volume confirmation (>2x 20-period average volume) 
and weekly pivot direction filter (price above/below weekly pivot) captures strong momentum moves. 
Weekly pivot provides structural bias: long only when price > weekly pivot, short only when price < weekly pivot. 
This avoids counter-trend breakouts that fail in ranging/bear markets. Uses 6h timeframe for optimal 
balance of signal quality and trade frequency. Target: 75-150 total trades over 4 years.
Works in bull/bear via weekly pivot filter that adapts to longer-term structure.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_breakout_12h_volume_weekly_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # === HTF: 12h data for volume spike filter ===
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    avg_vol_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = volume_12h > (2.0 * avg_vol_12h)
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h)
    
    # === HTF: 1d data for weekly pivot calculation ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate weekly pivot (using prior week's OHLC)
    # We'll approximate weekly by using 5 trading days (5x 1d bars)
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
    
    # === 6h Indicators ===
    # Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 6h volume average for confirmation
    avg_volume_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
            np.isnan(vol_spike_12h_aligned[i]) or np.isnan(weekly_pivot_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Weekly Pivot Direction Filter ---
        price_vs_pivot = close[i] - weekly_pivot_1d_aligned[i]
        bullish_bias = price_vs_pivot > 0   # Price above weekly pivot
        bearish_bias = price_vs_pivot < 0   # Price below weekly pivot
        
        # --- Donchian Breakout + Volume Confirmation ---
        # Upper breakout: price breaks above Donchian high with volume spike
        upper_breakout = (close[i] > donchian_high[i]) and vol_spike_12h_aligned[i]
        # Lower breakout: price breaks below Donchian low with volume spike
        lower_breakout = (close[i] < donchian_low[i]) and vol_spike_12h_aligned[i]
        
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
#!/usr/bin/env python3
"""
Experiment #265: 12h Donchian(20) Breakout + 1d EMA(50) Trend + Volume Spike Filter

HYPOTHESIS: Combining 12h Donchian channel breakouts with 1d EMA trend alignment and volume confirmation creates a robust trend-following strategy. The 1d EMA provides the primary trend direction, Donchian(20) breakouts on 12h timeframe capture momentum with structure, and volume spike filter ensures institutional participation. Targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag while capturing sustained moves in both bull and bear markets. Uses discrete position sizing (0.25) and ATR-based stoploss for risk management.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_ema_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(50) on 1d close
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)
    
    # === 12h Indicators ===
    # Donchian Channel (20-period)
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    
    if n >= 20:
        for i in range(n):
            start_idx = max(0, i - 19)
            donchian_upper[i] = np.max(high[start_idx:i+1])
            donchian_lower[i] = np.min(low[start_idx:i+1])
    
    # Volume Spike Detector (Volume > 2.0 * 20-period average)
    volume_ma = np.full(n, np.nan)
    if n >= 20:
        volume_series = pd.Series(volume)
        volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    volume_spike = np.zeros(n, dtype=bool)
    if n >= 20:
        volume_spike = volume > (2.0 * volume_ma)
        volume_spike[:20] = False  # First 20 bars invalid
    
    # ATR(14) for stoploss
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr_series = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean()
        atr = atr_series.values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * entry_atr
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * entry_atr
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Volume confirmation required
        if not volume_spike[i]:
            signals[i] = 0.0
            continue
        
        # Price above/below Donchian channels
        price_above_upper = close[i] > donchian_upper[i]
        price_below_lower = close[i] < donchian_lower[i]
        
        # Trend alignment with 1d EMA
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Long: Donchian breakout above upper + price above 1d EMA + volume spike
        if price_above_upper and price_above_ema:
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_atr = atr[i]
            signals[i] = SIZE
        # Short: Donchian breakout below lower + price below 1d EMA + volume spike
        elif price_below_lower and price_below_ema:
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_atr = atr[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals
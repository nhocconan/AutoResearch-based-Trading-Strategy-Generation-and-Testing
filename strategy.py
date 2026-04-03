#!/usr/bin/env python3
"""
Experiment #1933: 4h Donchian(20) Breakout + 12h HMA Trend + Volume Spike
HYPOTHESIS: Donchian channel breakouts capture institutional momentum. 
- Primary: 4h Donchian(20) breakout with volume confirmation (>2x 20-bar average)
- HTF: 12h HMA(21) for trend filter (only trade in direction of 12h trend)
- Exit: ATR-based stoploss (2*ATR) or time-based exit (max 10 bars)
- Works in bull/bear markets by following HTF trend with precise 4h entries.
Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1933_4h_donchian20_12h_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for HMA trend (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h HMA(21)
    def hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half = period // 2
        sqrt_period = int(np.sqrt(period))
        wma2 = pd.Series(arr).ewm(span=half, adjust=False).mean()
        wma1 = pd.Series(arr).ewm(span=period, adjust=False).mean()
        raw = 2 * wma2 - wma1
        hma_vals = pd.Series(raw).ewm(span=sqrt_period, adjust=False).mean()
        return hma_vals.values
    
    hma_12h = hma(close_12h, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # === 4h Indicators: Donchian(20), Volume MA(20), ATR(14) ===
    # Donchian channels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume MA for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # ATR for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for Donchian(20), ATR(14), volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(hma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Stoploss: 2*ATR below entry
                if price <= entry_price - 2.0 * entry_atr:
                    exit_signal = True
                # Time-based exit: max 10 bars
                elif bars_since_entry >= 10:
                    exit_signal = True
            else:  # Short position
                # Stoploss: 2*ATR above entry
                if price >= entry_price + 2.0 * entry_atr:
                    exit_signal = True
                # Time-based exit: max 10 bars
                elif bars_since_entry >= 10:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Trend filter: 12h HMA direction
        trend_up = hma_12h_aligned[i] > hma_12h_aligned[i-1] if i > 0 else False
        trend_down = hma_12h_aligned[i] < hma_12h_aligned[i-1] if i > 0 else False
        
        # Volume confirmation: require volume spike (> 2x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Long entry: price breaks above Donchian high AND 12h trend up
            if trend_up and price > donch_high[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian low AND 12h trend down
            elif trend_down and price < donch_low[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals
# Strategy: exp_266_4h_donchian_1d_hma_vol_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.008 | +18.7% | -29.0% | 20 | PASS |
| ETHUSDT | 0.217 | +33.6% | -21.5% | 55 | PASS |
| SOLUSDT | 1.151 | +259.2% | -27.1% | 57 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.213 | +9.2% | -19.5% | 18 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #266: 4h Donchian Breakout + 1d HMA Trend + Volume Spike
HYPOTHESIS: 4h Donchian(20) breakouts capture momentum, while 1d HMA(50) filters trend direction (bull/bear). 
Volume spike (>2x 20-period MA) confirms institutional participation. 
In bull market (price > 1d HMA): long on upper breakout, short on lower breakdown.
In bear market (price < 1d HMA): short on upper breakout (fade strength), long on lower breakdown (fade weakness).
ATR(14) stoploss (2.5x) manages risk. Discrete sizing 0.25 minimizes fee churn.
Target: 100-180 total trades over 4 years (25-45/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_266_4h_donchian_1d_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d HMA(50) for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    hma_50_1d = calculate_hma(close_1d, 50)
    hma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_50_1d)
    
    # === 4h Indicators: Donchian(20) channels ===
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: ATR(14) for stoploss and volatility filter ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # Enough for Donchian(20), ATR(14), volume MA(20)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(hma_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 2x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Trend Filter: 1d HMA(50) ---
        bull_trend = price > hma_50_1d_aligned[i]
        bear_trend = price < hma_50_1d_aligned[i]
        
        # --- Donchian Breakout Signals ---
        upper_breakout = price > highest_20[i-1]  # Break above prior 20-period high
        lower_breakout = price < lowest_20[i-1]   # Break below prior 20-period low
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        if volume_spike:
            if bull_trend:
                # Bull trend: follow breakouts
                if upper_breakout:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                elif lower_breakout:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
            elif bear_trend:
                # Bear trend: fade breakouts (contrarian)
                if upper_breakout:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                elif lower_breakout:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

def calculate_hma(arr, period):
    """Calculate Hull Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = np.zeros_like(arr)
    for i in range(len(arr)):
        if i < half_period - 1:
            wma_half[i] = np.nan
        else:
            start_idx = i - half_period + 1
            weights = np.arange(1, half_period + 1)
            wma_half[i] = np.dot(arr[start_idx:i+1], weights) / weights.sum()
    
    # WMA of full period
    wma_full = np.zeros_like(arr)
    for i in range(len(arr)):
        if i < period - 1:
            wma_full[i] = np.nan
        else:
            start_idx = i - period + 1
            weights = np.arange(1, period + 1)
            wma_full[i] = np.dot(arr[start_idx:i+1], weights) / weights.sum()
    
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final HMA: WMA(sqrt_period) of raw_hma
    hma = np.zeros_like(arr)
    for i in range(len(arr)):
        if i < sqrt_period - 1:
            hma[i] = np.nan
        else:
            start_idx = i - sqrt_period + 1
            weights = np.arange(1, sqrt_period + 1)
            hma[i] = np.dot(raw_hma[start_idx:i+1], weights) / weights.sum()
    
    return hma
```

## Last Updated
2026-04-03 14:07

# Strategy: exp_033_4h_donchian_12h_hma_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.908 | -0.7% | -3.3% | 61 | FAIL |
| ETHUSDT | 0.099 | +23.2% | -15.8% | 136 | PASS |
| SOLUSDT | 1.056 | +232.5% | -30.7% | 111 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.213 | +9.2% | -16.5% | 37 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #033: 4h Donchian Breakout + 12h HMA Trend + Volume Filter

HYPOTHESIS: Uses 4h Donchian channel breakouts (20-period) for entry timing,
filtered by 12h Hull Moving Average (21) trend direction and volume confirmation.
In strong trends (12h HMA slope aligned with breakout direction), we take
breakout pullbacks to the channel midpoint. Uses ATR-based stoploss (2.0) and
discrete position sizing (0.25) to minimize fee drag. Target: 75-200 trades over
4 years on 4h timeframe with proper risk control for both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_033_4h_donchian_12h_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for HMA trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    hma_12h = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 12h HMA slope for trend strength
    hma_slope = np.zeros_like(hma_12h_aligned)
    hma_slope[1:] = (hma_12h_aligned[1:] - hma_12h_aligned[:-1]) / hma_12h_aligned[:-1]
    hma_slope[0] = 0
    
    # === 4h Indicators: Donchian Channel (20) ===
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    donchian_high = rolling_max(high, 20)
    donchian_low = rolling_min(low, 20)
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # === 4h Indicators: ATR(14) for stoploss and volume filter ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Volume ratio: current volume / 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # Sufficient warmup for Donchian and ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_12h_aligned[i]) or np.isnan(hma_slope[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- 12h HMA Trend Filter ---
        # Strong uptrend: HMA rising (> 0.1% per bar)
        # Strong downtrend: HMA falling (< -0.1% per bar)
        # Neutral: |slope| <= 0.1%
        is_uptrend_12h = hma_slope[i] > 0.001
        is_downtrend_12h = hma_slope[i] < -0.001
        
        # --- Volume Confirmation ---
        # Require volume > 1.5x average for breakout validity
        vol_confirmed = vol_ratio[i] > 1.5
        
        # --- Breakout Detection ---
        # Bullish breakout: price closes above Donchian high
        # Bearish breakout: price closes below Donchian low
        bullish_breakout = close[i] > donchian_high[i-1]  # Close above prior high
        bearish_breakout = close[i] < donchian_low[i-1]   # Close below prior low
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit if trend reverses or volume dries up
                if not is_uptrend_12h or not vol_confirmed:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit if trend reverses or volume dries up
                if not is_downtrend_12h or not vol_confirmed:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = position_side * SIZE
                continue
            
            # Take profit at Donchian midpoint on reversal signs
            if position_side > 0 and price < donchian_mid[i] and hma_slope[i] < 0:
                signals[i] = 0.0  # Exit long at midpoint
                in_position = False
                position_side = 0
                bars_since_entry = 0
                continue
            elif position_side < 0 and price > donchian_mid[i] and hma_slope[i] > 0:
                signals[i] = 0.0  # Exit short at midpoint
                in_position = False
                position_side = 0
                bars_since_entry = 0
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Bullish breakout + 12h uptrend + volume confirmation
        if bullish_breakout and is_uptrend_12h and vol_confirmed:
            # Enter on retest of breakout level (donchian_high)
            if abs(price - donchian_high[i-1]) < 0.5 * atr_14[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
        # Short: Bearish breakout + 12h downtrend + volume confirmation
        elif bearish_breakout and is_downtrend_12h and vol_confirmed:
            # Enter on retest of breakout level (donchian_low)
            if abs(price - donchian_low[i-1]) < 0.5 * atr_14[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
    
    return signals

def calculate_hma(arr, period):
    """Hull Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=np.float64)
    
    # Calculate WMA for period/2
    half_period = max(1, period // 2)
    wma_half = np.zeros_like(arr, dtype=np.float64)
    for i in range(len(arr)):
        start_idx = max(0, i - half_period + 1)
        weights = np.arange(1, i - start_idx + 2)
        wma_half[i] = np.dot(arr[start_idx:i+1], weights) / weights.sum()
    
    # Calculate WMA for full period
    wma_full = np.zeros_like(arr, dtype=np.float64)
    for i in range(len(arr)):
        start_idx = max(0, i - period + 1)
        weights = np.arange(1, i - start_idx + 2)
        wma_full[i] = np.dot(arr[start_idx:i+1], weights) / weights.sum()
    
    # Calculate raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Calculate final WMA of raw HMA with sqrt(period)
    sqrt_period = max(1, int(np.sqrt(period)))
    hma = np.zeros_like(arr, dtype=np.float64)
    for i in range(len(arr)):
        start_idx = max(0, i - sqrt_period + 1)
        weights = np.arange(1, i - start_idx + 2)
        hma[i] = np.dot(raw_hma[start_idx:i+1], weights) / weights.sum()
    
    return hma
```

## Last Updated
2026-04-03 12:57

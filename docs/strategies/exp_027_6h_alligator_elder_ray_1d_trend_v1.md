# Strategy: exp_027_6h_alligator_elder_ray_1d_trend_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 1.650 | +71.5% | -2.9% | 431 | PASS |
| ETHUSDT | 2.215 | +119.1% | -4.5% | 415 | PASS |
| SOLUSDT | 2.733 | +249.6% | -3.4% | 403 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.980 | +10.9% | -2.2% | 117 | PASS |
| ETHUSDT | 2.590 | +30.8% | -2.0% | 104 | PASS |
| SOLUSDT | 1.785 | +22.4% | -2.9% | 126 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #027: 6h Williams Alligator + Elder Ray + 1d Trend Filter Strategy

HYPOTHESIS: Combines Williams Alligator (trend identification) with Elder Ray 
(Bull/Bear Power) on 6h timeframe, filtered by 1d trend direction. The Alligator 
identifies trending vs ranging markets via jaw-teeth-lips alignment, while Elder 
Ray measures bull/bear strength relative to EMA13. In strong trends (Alligator 
awake with Elder Ray confirmation), we enter pullbacks to the middle line (teeth). 
In ranging markets (Alligator sleeping), we avoid trades. Uses 1d trend filter 
to only trade in alignment with higher timeframe direction. Target: 75-150 trades 
over 4 years with discrete position sizing to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_027_6h_alligator_elder_ray_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 6h Indicators: Williams Alligator ===
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward  
    # Lips: 5-period SMMA, shifted 3 bars forward
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) >= period:
            # First value is simple SMA
            result[period-1] = np.mean(arr[:period])
            # Subsequent values: SMMA = (prev_SMMA*(period-1) + current_price) / period
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    # Shift forward as per Alligator definition
    jaw = np.roll(jaw, -8)
    teeth = np.roll(teeth, -5)
    lips = np.roll(lips, -3)
    # NaN out the shifted values at the end
    jaw[-8:] = np.nan
    teeth[-5:] = np.nan
    lips[-3:] = np.nan
    
    # === 6h Indicators: Elder Ray (Bull/Bear Power) ===
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # Sufficient warmup for SMMA and EMA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- 1d Trend Filter: Only trade when price is clearly above/below EMA50 ---
        is_uptrend_1d = price > ema50_1d_aligned[i] * 1.002
        is_downtrend_1d = price < ema50_1d_aligned[i] * 0.998
        
        # --- Williams Alligator Conditions ---
        # Alligator awake: lips > teeth > jaw (uptrend) OR lips < teeth < jaw (downtrend)
        alligator_awake_up = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_awake_down = lips[i] < teeth[i] and teeth[i] < jaw[i]
        alligator_sleeping = not (alligator_awake_up or alligator_awake_down)
        
        # --- Elder Ray Conditions ---
        # Strong bull power: positive and increasing
        # Strong bear power: negative and decreasing (more negative)
        bull_strong = bull_power[i] > 0 and bull_power[i] > bull_power[i-1]
        bear_strong = bear_power[i] < 0 and bear_power[i] < bear_power[i-1]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit if Alligator starts sleeping or Elder Ray weakens
                if alligator_sleeping or not bull_strong:
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
                # Exit if Alligator starts sleeping or Elder Ray weakens
                if alligator_sleeping or not bear_strong:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 4 bars to reduce churn
            if bars_since_entry < 4:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Only trade when Alligator is awake AND aligned with 1d trend
        if not alligator_sleeping:
            # Long: Alligator awake up AND 1d uptrend AND strong bull power
            if alligator_awake_up and is_uptrend_1d and bull_strong:
                # Enter on pullback to teeth (red line) - wait for price to touch/near teeth
                if abs(price - teeth[i]) < 0.5 * atr_14[i]:  # Near teeth
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
            # Short: Alligator awake down AND 1d downtrend AND strong bear power
            elif alligator_awake_down and is_downtrend_1d and bear_strong:
                # Enter on pullback to teeth (red line)
                if abs(price - teeth[i]) < 0.5 * atr_14[i]:  # Near teeth
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
    
    return signals
```

## Last Updated
2026-04-03 12:55

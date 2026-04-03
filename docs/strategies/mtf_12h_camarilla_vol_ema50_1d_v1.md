# Strategy: mtf_12h_camarilla_vol_ema50_1d_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.142 | +18.0% | -5.8% | 312 | FAIL |
| ETHUSDT | 0.076 | +23.5% | -7.7% | 262 | PASS |
| SOLUSDT | 0.515 | +52.4% | -12.4% | 250 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.394 | +10.0% | -4.7% | 93 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #005: 12h Camarilla Pivot + Volume + 1d EMA50 Trend

HYPOTHESIS: Camarilla S3/S4 and R3/R4 are institutional support/resistance levels.
By entering when price touches these levels WITH volume confirmation AND 1d trend
alignment, this strategy catches reversals at key levels.

WHY 12h: 3x slower than 4h = ~3x fewer trades = less fee drag.
Camarilla levels on 12h capture multi-day swings.

WHY IT WORKS IN BULL AND BEAR: Symmetrical pivot levels — buy S3/S4 in uptrends,
short R3/R4 in downtrends. Works in both bull rallies and bear bounces.

TARGET: 75-150 total trades over 4 years = 19-37/year. HARD MAX: 200.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_vol_ema50_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend direction (faster than 200 for more signals)
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 100  # Need enough for EMA50 alignment buffer
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if EMA not aligned
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA50) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === CAMARILLA LEVELS from previous CLOSED bar (no look-ahead) ===
        prev_high = high[i - 1]
        prev_low = low[i - 1]
        prev_close = close[i - 1]
        prev_range = prev_high - prev_low
        
        # Classic Camarilla levels (factor 1.1/12 = 0.09167)
        r3 = prev_close + prev_range * 0.09167
        r4 = prev_close + prev_range * 0.18333
        s3 = prev_close - prev_range * 0.09167
        s4 = prev_close - prev_range * 0.18333
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Price touches S3 or S4 with volume + trend alignment ===
            if price_above_1d_ema and vol_spike:
                # S4 touch (deeper level = better risk/reward)
                if low[i] <= s4:
                    desired_signal = SIZE
                # S3 touch (softer level)
                elif low[i] <= s3:
                    desired_signal = SIZE
            
            # === SHORT: Price touches R3 or R4 with volume + trend alignment ===
            if not price_above_1d_ema and vol_spike:
                # R4 touch
                if high[i] >= r4:
                    desired_signal = -SIZE
                # R3 touch
                elif high[i] >= r3:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === HOLD PERIOD (minimum 2 bars = 1 day to avoid churn) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 2:
            # Take profit if price reverts to Camarilla mid (prev close)
            if position_side > 0 and close[i] >= prev_close:
                desired_signal = 0.0
            if position_side < 0 and close[i] <= prev_close:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals
```

## Last Updated
2026-03-30 02:43

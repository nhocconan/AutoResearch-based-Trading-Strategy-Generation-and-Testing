# Strategy: mtf_4h_donchian_1d_hma_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.273 | +35.3% | -13.8% | 146 | PASS |
| ETHUSDT | 0.578 | +65.7% | -12.8% | 141 | PASS |
| SOLUSDT | 0.546 | +83.4% | -23.2% | 134 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | -0.061 | +3.5% | -10.6% | 52 | FAIL |
| SOLUSDT | 0.428 | +14.1% | -18.3% | 39 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #009: 4h Donchian(20) Breakout + 1d HMA Trend + Volume Confirmation + ATR Stoploss

HYPOTHESIS: Combining a price channel breakout (Donchian 20) with HTF trend confirmation (1d HMA) 
and volume spike creates a high-probability entry signal. The strategy only trades in the direction 
of the 1d trend, reducing whipsaws in ranging markets. ATR-based trailing stoploss manages risk. 
Designed to capture medium-term trends in both bull and bear markets with controlled trade frequency 
(~19-50 trades/year) to minimize fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_1d_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(data, window):
        weights = np.arange(1, window + 1, dtype=np.float64)
        return pd.Series(data).rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        ).values
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss calculation."""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d HMA for trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # === 4h Indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete position sizing (30% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 50  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if np.isnan(atr_14[i]) or np.isnan(hma_1d_aligned[i]) or np.isnan(dc_upper_20[i]) or np.isnan(dc_lower_20[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # --- HTF Trend Check ---
        htf_bullish = close[i] > hma_1d_aligned[i]
        htf_bearish = close[i] < hma_1d_aligned[i]
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 2.0 if vol_ma_20[i] > 1e-10 else False  # Strict 2x volume spike
        
        # --- Position Management (Exit Logic) ---
        stop_hit = False
        
        if in_position:
            # ATR-based trailing stoploss
            if position_side > 0:
                stop_level = highest_since_entry - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    stop_hit = True
            else:  # Short position
                stop_level = lowest_since_entry + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    stop_hit = True
            
            # Trend reversal exit (only after minimum hold to avoid premature exits)
            min_hold = (i - entry_bar) >= 4
            if min_hold:
                if position_side > 0 and htf_bearish:
                    stop_hit = True
                if position_side < 0 and htf_bullish:
                    stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Require ALL conditions: HTF trend + breakout + volume confirmation
        if htf_bullish and bullish_breakout and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        elif htf_bearish and bearish_breakout and vol_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals
```

## Last Updated
2026-04-03 07:22

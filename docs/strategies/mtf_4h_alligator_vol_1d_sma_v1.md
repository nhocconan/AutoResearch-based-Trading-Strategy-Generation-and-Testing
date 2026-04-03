# Strategy: mtf_4h_alligator_vol_1d_sma_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.066 | +22.9% | -11.2% | 211 | PASS |
| ETHUSDT | 0.053 | +21.7% | -15.0% | 211 | PASS |
| SOLUSDT | 0.750 | +104.8% | -24.7% | 207 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.032 | +5.6% | -12.6% | 72 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #022: 4h Williams Alligator + Volume Spike + 1d SMA Filter

HYPOTHESIS: Williams Alligator is a multi-SMA trend detection system that
captures market structure without repainting. The Lips crossing above/below
the Jaw provides clean trend signals. Combined with volume confirmation and
1d SMA(50) for macro trend filter, this should:
- Work in 2021 bull:捕捉動能突破
- Work in 2022 bear: 空頭排列時做空
- Work in 2025 range: Macro filter keeps flat in chop

KEY INSIGHT FROM DB: Winning strategies use ONE strong signal (price channel 
or momentum oscillator) + volume confirmation + regime filter. Alligator 
provides BOTH trend direction AND regime (awake/sleeping) information.

TRADE COUNT: 75-200 total over 4 years (target 20-50/year).
Size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_alligator_vol_1d_sma_v1"
timeframe = "4h"
leverage = 1.0

def calculate_alligator(high, low, close, period=13):
    """
    Williams Alligator indicator.
    Jaw = SMA(median, 13) smoothed by SMA(8)
    Teeth = SMA(median, 8) smoothed by SMA(5)
    Lips = SMA(median, 5) smoothed by SMA(3)
    """
    median = (high + low + close) / 3.0
    
    # Base SMAs
    jaw_base = pd.Series(median).rolling(window=13, min_periods=13).mean().values
    teeth_base = pd.Series(median).rolling(window=8, min_periods=8).mean().values
    lips_base = pd.Series(median).rolling(window=5, min_periods=5).mean().values
    
    # Smoothed values (SMMA-like using ewm for efficiency)
    jaw = pd.Series(jaw_base).ewm(span=8, min_periods=8, adjust=False).mean().values
    teeth = pd.Series(teeth_base).ewm(span=5, min_periods=5, adjust=False).mean().values
    lips = pd.Series(lips_base).ewm(span=3, min_periods=3, adjust=False).mean().values
    
    return jaw, teeth, lips

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
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
    
    # === HTF: 1d SMA for macro trend (call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    sma_1d_50 = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_50)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Williams Alligator
    jaw, teeth, lips = calculate_alligator(high, low, close, period=13)
    
    # Volume spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 60  # Alligator needs time to stabilize
    
    for i in range(warmup, n):
        # NaN check
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Update highest/lowest for trailing stop
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === ALLIGATOR SIGNALS ===
        # Lips crosses above Teeth crosses above Jaw = bullish (alligator eating)
        alligator_bullish = lips[i] > teeth[i] > jaw[i]
        # Lips crosses below Teeth crosses below Jaw = bearish (alligator eating)
        alligator_bearish = lips[i] < teeth[i] < jaw[i]
        
        # Alligator awake (spread between lines > threshold = trending)
        alligator_spread = abs(lips[i] - jaw[i]) / jaw[i] if jaw[i] != 0 else 0
        alligator_awake = alligator_spread > 0.005  # 0.5% minimum spread
        
        # === HTF MACRO FILTER (1d SMA) ===
        htf_bullish = close[i] > sma_1d_aligned[i]
        htf_bearish = close[i] < sma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION (>1.5x average) ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === MINIMUM HOLD: 3 bars (12h) to avoid immediate reversals ===
        min_hold_bars = 3
        min_hold = (i - entry_bar) >= min_hold_bars
        
        # === ATR TRAILING STOP (2.5x ATR from entry high/low) ===
        def check_atr_stop():
            if not in_position:
                return False
            if position_side > 0:
                return low[i] < (highest_since_entry - 2.5 * entry_atr)
            else:
                return high[i] > (lowest_since_entry + 2.5 * entry_atr)
        
        # === EXITS ===
        if in_position:
            stop_hit = check_atr_stop()
            
            # Exit on trend reversal with min hold
            if position_side > 0 and alligator_bearish and min_hold:
                stop_hit = True
            if position_side < 0 and alligator_bullish and min_hold:
                stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # LONG: Alligator bullish + volume spike + HTF bullish + trending
            if alligator_bullish and vol_spike and htf_bullish and alligator_awake:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            # SHORT: Alligator bearish + volume spike + HTF bearish + trending
            elif alligator_bearish and vol_spike and htf_bearish and alligator_awake:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            
            else:
                signals[i] = 0.0
    
    return signals
```

## Last Updated
2026-03-30 12:58

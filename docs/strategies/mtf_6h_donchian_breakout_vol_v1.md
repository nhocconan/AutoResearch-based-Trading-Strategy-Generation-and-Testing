# Strategy: mtf_6h_donchian_breakout_vol_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.150 | +14.2% | -12.4% | 135 | FAIL |
| ETHUSDT | -0.232 | +6.9% | -13.5% | 149 | FAIL |
| SOLUSDT | 0.414 | +55.5% | -22.9% | 142 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.809 | +19.3% | -7.6% | 43 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #024: 6h Donchian Breakout + Volume + ATR Trailing Stop

HYPOTHESIS: Simple 6h Donchian(20) breakout with volume confirmation is a proven
edge from 16K+ experiments. 6h is optimal for trade frequency (between 4h and 12h).
- 2021 bull: breakout above Donchian = trend continuation
- 2022 bear: breakdown below Donchian = short rallies  
- 2025 range: choppy, fewer signals = adaptive

KEY INSIGHT: Keep it SIMPLE. 2-3 conditions. Previous failures had too many
filters → 0 trades. This loosens entry to hit 75-150 trades target.

TARGET: 75-150 total over 4 years (18-37/year). Size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_breakout_vol_v1"
timeframe = "6h"
leverage = 1.0

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
    
    # === HTF: 1d SMA for macro direction (call ONCE) ===
    df_1d = get_htf_data(prices, '1d')
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # === Indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian 20 - breakout channel
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation (20-bar average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 50
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === HTF TREND ===
        htf_bullish = close[i] > sma_1d_aligned[i]
        htf_bearish = close[i] < sma_1d_aligned[i]
        
        # === BREAKOUT CONDITIONS ===
        # Price breaks above 20-bar high = bullish breakout
        bullish_breakout = close[i] > dc_upper_20[i] if not np.isnan(dc_upper_20[i]) else False
        # Price breaks below 20-bar low = bearish breakout
        bearish_breakout = close[i] < dc_lower_20[i] if not np.isnan(dc_lower_20[i]) else False
        
        # === VOLUME CONFIRMATION ===
        vol_ok = volume[i] > vol_ma[i] * 1.3 if vol_ma[i] > 1e-10 else False
        
        # Update highest/lowest for trailing stop
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === MIN HOLD: 2 bars (12h) ===
        min_hold = (i - entry_bar) >= 2
        
        # === ATR TRAILING STOP (2.0x ATR from highest/lowest) ===
        if in_position:
            stop_hit = False
            if position_side > 0:
                # Long stop: price drops below highest - 2*ATR
                stop_hit = low[i] < (highest_since_entry - 2.0 * atr_14[i])
            else:
                # Short stop: price rises above lowest + 2*ATR
                stop_hit = high[i] > (lowest_since_entry + 2.0 * atr_14[i])
            
            # Exit on opposite HTF trend (after min hold)
            if min_hold:
                if position_side > 0 and htf_bearish:
                    stop_hit = True
                if position_side < 0 and htf_bullish:
                    stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # LONG: Breakout above + volume confirm + 1d uptrend
            if bullish_breakout and vol_ok and htf_bullish:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            # SHORT: Breakdown below + volume confirm + 1d downtrend
            elif bearish_breakout and vol_ok and htf_bearish:
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
2026-03-30 13:45

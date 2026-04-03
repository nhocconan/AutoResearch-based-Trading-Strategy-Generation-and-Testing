# Strategy: mtf_6h_donchian_weekly_pivot_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.398 | +11.2% | -8.2% | 93 | FAIL |
| ETHUSDT | 0.584 | +45.0% | -5.9% | 88 | PASS |
| SOLUSDT | 0.251 | +35.2% | -18.6% | 73 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.714 | +14.2% | -6.2% | 34 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #255: 6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation

HYPOTHESIS: 6h Donchian breakouts aligned with weekly Camarilla pivot levels (R4/S4) capture 
strong momentum with institutional participation. Weekly pivot provides structural support/resistance 
from higher timeframe, reducing false breakouts. Volume confirmation (2.0x average) ensures 
follow-through. Designed for 6h timeframe to target 12-37 trades/year (50-150 over 4 years).
Works in both bull and bear markets by only taking breakouts in direction of weekly pivot bias.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly Camarilla pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly Camarilla pivot levels from prior week's daily OHLC
    # We'll use rolling window of 5 days (1 week) to get weekly high/low/close
    if len(df_1d) >= 5:
        # Weekly high, low, close from prior completed week
        weekly_high = pd.Series(df_1d['high'].values).rolling(window=5, min_periods=5).max().shift(1).values
        weekly_low = pd.Series(df_1d['low'].values).rolling(window=5, min_periods=5).min().shift(1).values
        weekly_close = pd.Series(df_1d['close'].values).rolling(window=5, min_periods=5).last().shift(1).values
        
        # Camarilla pivot levels
        weekly_range = weekly_high - weekly_low
        # R4 = Close + Range * 1.1/2
        r4 = weekly_close + weekly_range * 1.1 / 2
        # S4 = Close - Range * 1.1/2
        s4 = weekly_close - weekly_range * 1.1 / 2
        
        # Align to 6h timeframe
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
        
        # Weekly bias: 1 if price > midpoint (bullish), -1 if price < midpoint (bearish)
        weekly_midpoint = (r4_aligned + s4_aligned) / 2
        weekly_bias = np.where(close[:len(r4_aligned)] > weekly_midpoint, 1, -1)
    else:
        r4_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
        weekly_bias = np.full(n, 0)
    
    # === 6h Indicators ===
    # ATR(14) for stoploss
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Donchian Channel(20) - shift(1) to avoid look-ahead
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume MA(20) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(dc_upper_20[i]) or np.isnan(dc_lower_20[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(r4_aligned[i]) if i < len(r4_aligned) else True or
            np.isnan(s4_aligned[i]) if i < len(s4_aligned) else True or i >= len(weekly_bias)):
            signals[i] = 0.0
            continue
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 2.0 if vol_ma_20[i] > 1e-10 else False  # 2.0x volume spike
        
        # --- Weekly Pivot Levels and Bias ---
        # Only trade breakouts that align with weekly bias AND break beyond R4/S4
        bullish_aligned = bullish_breakout and weekly_bias[i] > 0 and close[i] > r4_aligned[i]
        bearish_aligned = bearish_breakout and weekly_bias[i] < 0 and close[i] < s4_aligned[i]
        
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
            
            # Exit conditions: trend reversal or opposite Donchian touch
            min_hold = (i - entry_bar) >= 3  # Minimum 3 bars hold (~18h)
            if min_hold:
                if position_side > 0:
                    # Exit long: price touches lower Donchian OR weekly bias turns bearish
                    if close[i] <= dc_lower_20[i] or weekly_bias[i] < 0:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price touches upper Donchian OR weekly bias turns bullish
                    if close[i] >= dc_upper_20[i] or weekly_bias[i] > 0:
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
        # Long conditions: 
        # Breakout above upper Donchian with volume confirmation, weekly bias bullish, and price > R4
        if bullish_aligned and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with volume confirmation, weekly bias bearish, and price < S4
        elif bearish_aligned and vol_ok:
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
2026-04-03 08:42

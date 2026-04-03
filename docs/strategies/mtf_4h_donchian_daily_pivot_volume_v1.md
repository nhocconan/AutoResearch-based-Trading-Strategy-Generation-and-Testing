# Strategy: mtf_4h_donchian_daily_pivot_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.447 | +0.1% | -16.8% | 165 | FAIL |
| ETHUSDT | 0.216 | +31.8% | -11.9% | 151 | PASS |
| SOLUSDT | 0.793 | +111.8% | -17.8% | 138 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.211 | +8.9% | -11.4% | 44 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #081: 4h Donchian(20) Breakout + Daily Pivot Direction + Volume Spike

HYPOTHESIS: 4h Donchian breakouts aligned with daily pivot levels (from 1d HTF) capture institutional order flow.
Daily pivot provides structural support/resistance from prior day's range. Volume confirmation (2.0x average) ensures
follow-through. Designed for 19-50 trades/year to minimize fee drag while maintaining statistical significance.
Uses discrete position sizing (0.25) to reduce churn. Works in both bull/bear markets by trading breakouts
in direction of daily pivot bias.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_daily_pivot_volume_v1"
timeframe = "4h"
leverage = 1.0

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

def calculate_daily_pivot(df_daily):
    """Calculate daily pivot levels from daily OHLC data.
    Uses prior day's high, low, close to compute:
    PP = (H + L + C) / 3
    R1 = 2*PP - L, S1 = 2*PP - H
    R2 = PP + (H - L), S2 = PP - (H - L)
    R3 = H + 2*(PP - L), S3 = L - 2*(H - PP)
    """
    n = len(df_daily)
    if n < 2:  # Need at least 2 days of data
        return {
            'PP': np.full(n, np.nan),
            'R1': np.full(n, np.nan), 'S1': np.full(n, np.nan),
            'R2': np.full(n, np.nan), 'S2': np.full(n, np.nan),
            'R3': np.full(n, np.nan), 'S3': np.full(n, np.nan)
        }
    
    # Shift by 1 to use prior day's OHLC
    high = pd.Series(df_daily['high'].values).shift(1)
    low = pd.Series(df_daily['low'].values).shift(1)
    close = pd.Series(df_daily['close'].values).shift(1)
    
    # Calculate pivot points
    PP = (high + low + close) / 3.0
    R1 = 2 * PP - low
    S1 = 2 * PP - high
    R2 = PP + (high - low)
    S2 = PP - (high - low)
    R3 = high + 2 * (PP - low)
    S3 = low - 2 * (high - PP)
    
    return {
        'PP': PP.values,
        'R1': R1.values, 'S1': S1.values,
        'R2': R2.values, 'S2': S2.values,
        'R3': R3.values, 'S3': S3.values
    }

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for daily pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    pivot_data = calculate_daily_pivot(df_1d)
    
    # Align all pivot levels to LTF
    PP_aligned = align_htf_to_ltf(prices, df_1d, pivot_data['PP'])
    R1_aligned = align_htf_to_ltf(prices, df_1d, pivot_data['R1'])
    S1_aligned = align_htf_to_ltf(prices, df_1d, pivot_data['S1'])
    R2_aligned = align_htf_to_ltf(prices, df_1d, pivot_data['R2'])
    S2_aligned = align_htf_to_ltf(prices, df_1d, pivot_data['S2'])
    R3_aligned = align_htf_to_ltf(prices, df_1d, pivot_data['R3'])
    S3_aligned = align_htf_to_ltf(prices, df_1d, pivot_data['S3'])
    
    # === 4h Indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
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
    
    warmup = 50  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(dc_upper_20[i]) or np.isnan(dc_lower_20[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(PP_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(R2_aligned[i]) or np.isnan(S2_aligned[i]) or
            np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Daily Pivot Bias ---
        # Price above daily pivot = bullish bias, below = bearish bias
        pivot_bullish = close[i] > PP_aligned[i]
        pivot_bearish = close[i] < PP_aligned[i]
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 2.0 if vol_ma_20[i] > 1e-10 else False  # 2.0x volume spike
        
        # --- Position Management (Exit Logic) ---
        stop_hit = False
        
        if in_position:
            # ATR-based trailing stoploss
            if position_side > 0:
                stop_level = highest_since_entry - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    stop_hit = True
            else:  # Short position
                stop_level = lowest_since_entry + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    stop_hit = True
            
            # Exit conditions: trend reversal or opposite Donchian touch
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold (~8h)
            if min_hold:
                if position_side > 0:
                    # Exit long: price touches lower Donchian OR breaks below S1
                    if close[i] <= dc_lower_20[i] or close[i] < S1_aligned[i]:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price touches upper Donchian OR breaks above R1
                    if close[i] >= dc_upper_20[i] or close[i] > R1_aligned[i]:
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
        # Breakout above upper Donchian with bullish daily pivot bias and volume confirmation
        if bullish_breakout and pivot_bullish and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with bearish daily pivot bias and volume confirmation
        elif bearish_breakout and pivot_bearish and vol_ok:
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
2026-04-03 07:49

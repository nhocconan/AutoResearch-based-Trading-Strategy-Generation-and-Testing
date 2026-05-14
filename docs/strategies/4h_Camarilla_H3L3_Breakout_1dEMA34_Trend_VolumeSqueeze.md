# Strategy: 4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSqueeze

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.361 | +31.5% | -5.0% | 261 | PASS |
| ETHUSDT | 0.056 | +22.8% | -10.1% | 250 | PASS |
| SOLUSDT | 0.485 | +48.8% | -11.1% | 221 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.024 | -6.1% | -7.8% | 105 | FAIL |
| ETHUSDT | 0.118 | +7.0% | -8.5% | 89 | PASS |
| SOLUSDT | 0.645 | +11.6% | -4.2% | 73 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSqueeze
Hypothesis: Camarilla H3/L3 breakouts with volume squeeze breakout and 1d EMA34 trend filter capture explosive moves.
Volume squeeze (low volatility) followed by expansion captures institutional accumulation/distribution.
Trades only in 1d EMA34 trend direction to avoid counter-trend whipsaw. Targets 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average with min_periods"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate Average True Range with min_periods"""
    if len(high) < period:
        return np.full_like(high, np.nan)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the period"""
    # Camarilla equations
    pivot = (high + low + close) / 3
    range_hl = high - low
    
    # Resistance levels
    r1 = close + (range_hl * 1.1 / 12)
    r2 = close + (range_hl * 1.1 / 6)
    r3 = close + (range_hl * 1.1 / 4)
    r4 = close + (range_hl * 1.1 / 2)
    
    # Support levels
    s1 = close - (range_hl * 1.1 / 12)
    s2 = close - (range_hl * 1.1 / 6)
    s3 = close - (range_hl * 1.1 / 4)
    s4 = close - (range_hl * 1.1 / 2)
    
    return r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for EMA34 trend filter and ATR (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 trend filter
    ema_34_1d = calculate_ema(df_1d['close'].values, 34)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d ATR for stoploss
    atr_1d = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Camarilla levels from 1d data (for 4h breakout signals)
    r1, r2, r3, r4, s1, s2, s3, s4 = calculate_camarilla(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values
    )
    h3_aligned = align_htf_to_ltf(prices, df_1d, r3)  # H3 = R3
    l3_aligned = align_htf_to_ltf(prices, df_1d, s3)  # L3 = S3
    
    # Volume squeeze: current volume < 0.5 * 20-period average (low volatility)
    # Volume expansion: current volume > 2.0 * 20-period average (breakout)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_squeeze = volume < (vol_ma * 0.5)
    volume_expansion = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # Track entry price for ATR-based stoploss
    
    # Start index: need enough for EMA (34) + volume MA (20) + ATR (14)
    start_idx = max(34, 20, 14) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla H3/L3 breakout + volume expansion + 1d EMA34 trend alignment
            # Only enter after a volume squeeze period (low volatility) to catch breakouts
            long_entry = (curr_close > h3_aligned[i]) and volume_expansion[i] and (curr_close > ema_34_1d_aligned[i])
            short_entry = (curr_close < l3_aligned[i]) and volume_expansion[i] and (curr_close < ema_34_1d_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below H3, trend turns bearish, or ATR stoploss hit
            atr_stop = entry_price - (1.5 * atr_1d_aligned[i])
            if curr_close < h3_aligned[i] or curr_close < ema_34_1d_aligned[i] or curr_close < atr_stop:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above L3, trend turns bullish, or ATR stoploss hit
            atr_stop = entry_price + (1.5 * atr_1d_aligned[i])
            if curr_close > l3_aligned[i] or curr_close > ema_34_1d_aligned[i] or curr_close > atr_stop:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSqueeze"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 08:42

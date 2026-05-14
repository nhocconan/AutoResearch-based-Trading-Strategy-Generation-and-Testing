# Strategy: 4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_ATRStop

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.814 | +52.7% | -7.1% | 272 | PASS |
| ETHUSDT | 0.537 | +45.7% | -7.7% | 247 | PASS |
| SOLUSDT | 0.513 | +57.7% | -19.2% | 202 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.941 | -7.6% | -7.9% | 106 | FAIL |
| ETHUSDT | 1.114 | +20.0% | -8.7% | 87 | PASS |
| SOLUSDT | 0.910 | +17.2% | -6.2% | 68 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_ATRStop
Hypothesis: Camarilla R3/S3 breakouts with volume spike confirmation and 1d EMA34 trend filter, combined with ATR-based stop loss, capture institutional order flow while managing downside risk. The ATR stop reduces whipsaws in choppy markets and improves risk-adjusted returns. Target trade frequency: 20-40/year to minimize fee drag. Works in both bull (breakouts) and bear (mean reversion at R3/S3) markets.
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
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots, EMA34 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 trend filter
    ema_34_1d = calculate_ema(df_1d['close'].values, 34)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d ATR for stop loss
    atr_1d = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1d Camarilla levels (based on previous day's OHLC)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels: R3, S3
    camarilla_range = 1.1 * (prev_high - prev_low)
    r3 = prev_close + camarilla_range * 0.55
    s3 = prev_close - camarilla_range * 0.55
    
    # Align Camarilla levels to 4h timeframe (already completed 1d bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA (34) + volume MA (20) + ATR (14) + Camarilla (2)
    start_idx = max(34, 20, 14, 2)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla R3/S3 breakout + volume spike + 1d EMA34 trend alignment
            long_breakout = curr_high > r3_aligned[i]
            short_breakout = curr_low < s3_aligned[i]
            
            long_entry = long_breakout and volume_spike[i] and (curr_close > ema_34_1d_aligned[i])
            short_entry = short_breakout and volume_spike[i] and (curr_close < ema_34_1d_aligned[i])
            
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
            # Long position: exit when price closes below R3 (failed breakout) or trend turns bearish or ATR stop hit
            atr_stop = entry_price - (1.5 * atr_1d_aligned[i])
            if curr_close < r3_aligned[i] or curr_close < ema_34_1d_aligned[i] or curr_close < atr_stop:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above S3 (failed breakout) or trend turns bullish or ATR stop hit
            atr_stop = entry_price + (1.5 * atr_1d_aligned[i])
            if curr_close > s3_aligned[i] or curr_close > ema_34_1d_aligned[i] or curr_close > atr_stop:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 08:44

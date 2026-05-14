# Strategy: 4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_RegimeFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.422 | +40.0% | -9.3% | 257 | PASS |
| ETHUSDT | 0.113 | +25.3% | -13.7% | 245 | PASS |
| SOLUSDT | 0.551 | +71.2% | -26.8% | 197 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.160 | -4.7% | -7.8% | 95 | FAIL |
| ETHUSDT | 1.643 | +34.2% | -7.2% | 79 | PASS |
| SOLUSDT | 0.157 | +7.8% | -8.6% | 66 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_RegimeFilter
Hypothesis: Camarilla R3/S3 breakouts with 1d EMA34 trend filter and 4h ADX regime filter.
ADX > 25 confirms strong trend for breakout continuation; ADX < 20 avoids whipsaw in ranging markets.
Volume spike confirms institutional participation. Discrete sizing (0.30) limits fee drag.
Designed to work in bull (trend continuation) and bear (mean reversion at R3/S3) markets.
Target: 30-60 trades/year to stay within proven winning range.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average with min_periods"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index with min_periods"""
    if len(high) < period + 1:
        return np.full_like(high, np.nan)
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=period, adjust=False, min_periods=period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / np.where(atr != 0, atr, np.nan)
    di_minus = 100 * dm_minus_smooth / np.where(atr != 0, atr, np.nan)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) != 0, (di_plus + di_minus), np.nan)
    adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots and EMA34 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 trend filter
    ema_34_1d = calculate_ema(df_1d['close'].values, 34)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d Camarilla levels (based on previous day's OHLC) - using R3/S3 (tighter bands for precision)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels: R3, S3 (standard bands)
    camarilla_range = 1.1 * (prev_high - prev_low)
    r3 = prev_close + camarilla_range * 0.25  # R3 level
    s3 = prev_close - camarilla_range * 0.25  # S3 level
    
    # Align Camarilla levels to 4h timeframe (already completed 1d bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 4h ADX for regime filter (trending vs ranging)
    adx = calculate_adx(high, low, close, 14)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA (34) + volume MA (20) + ADX (14) + Camarilla (2)
    start_idx = max(34, 20, 14, 2)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla R3/S3 breakout + volume spike + 1d EMA34 trend alignment + ADX > 25 (trending)
            long_breakout = curr_high > r3_aligned[i]
            short_breakout = curr_low < s3_aligned[i]
            
            long_entry = (long_breakout and volume_spike[i] and 
                         (curr_close > ema_34_1d_aligned[i]) and (adx[i] > 25))
            short_entry = (short_breakout and volume_spike[i] and 
                          (curr_close < ema_34_1d_aligned[i]) and (adx[i] > 25))
            
            if long_entry:
                signals[i] = 0.30
                position = 1
            elif short_entry:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below R3 (failed breakout) or trend turns bearish or ADX < 20 (ranging)
            if curr_close < r3_aligned[i] or curr_close < ema_34_1d_aligned[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position: exit when price closes above S3 (failed breakout) or trend turns bullish or ADX < 20 (ranging)
            if curr_close > s3_aligned[i] or curr_close > ema_34_1d_aligned[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_RegimeFilter"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 08:51

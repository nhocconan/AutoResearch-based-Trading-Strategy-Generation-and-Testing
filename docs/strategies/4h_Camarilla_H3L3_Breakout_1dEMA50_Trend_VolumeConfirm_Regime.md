# Strategy: 4h_Camarilla_H3L3_Breakout_1dEMA50_Trend_VolumeConfirm_Regime

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.495 | +44.8% | -9.5% | 337 | KEEP |
| ETHUSDT | -0.040 | +16.6% | -13.0% | 329 | DISCARD |
| SOLUSDT | 0.763 | +102.9% | -22.1% | 288 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.117 | +7.0% | -4.2% | 116 | KEEP |
| SOLUSDT | 1.284 | +28.4% | -7.6% | 99 | KEEP |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_H3L3_Breakout_1dEMA50_Trend_VolumeConfirm_Regime
Hypothesis: Trade 4h Camarilla H3/L3 breakouts with 1d EMA50 trend filter, volume confirmation (>1.5x 20-bar MA), and choppiness regime filter (CHOP < 61.8 for trending markets). 
This strategy targets trending markets only to avoid false breakouts in ranging conditions, reducing whipsaws and improving signal quality. 
Discrete sizing 0.25 balances profit and fee drag. Target: 20-35 trades/year (~80-140 over 4 years) to stay within fee drag limits.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous 1d bar's OHLC
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    prev_close_1d = df_1d['close'].shift(1).values
    
    camarilla_range = prev_high_1d - prev_low_1d
    h3 = prev_close_1d + 1.1 * camarilla_range / 6   # H3 level
    l3 = prev_close_1d - 1.1 * camarilla_range / 6   # L3 level
    
    # Align Camarilla levels to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # Choppiness regime filter: CHOP < 61.8 indicates trending market (use 14-period)
    # True Range = max(high-low, abs(high-previous_close), abs(low-previous_close))
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Sum of absolute price changes over 14 periods
    abs_changes = np.abs(np.diff(close, prepend=close[0]))
    sum_abs_changes = pd.Series(abs_changes).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_abs_changes / (atr_14 * 14)) / np.log10(10)
    chop_regime = chop < 61.8  # Trending market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 1d EMA50 (50), volume MA (20), and CHOP (14)
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above H3 AND 1d trend bullish (close > EMA50) AND volume confirm AND trending regime
            long_setup = (close[i] > h3_aligned[i]) and \
                         (close[i] > ema_50_1d_aligned[i]) and \
                         volume_confirm[i] and \
                         chop_regime[i]
            # Short: price breaks below L3 AND 1d trend bearish (close < EMA50) AND volume confirm AND trending regime
            short_setup = (close[i] < l3_aligned[i]) and \
                          (close[i] < ema_50_1d_aligned[i]) and \
                          volume_confirm[i] and \
                          chop_regime[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price re-enters Camarilla H3/L3 range OR 1d trend turns bearish OR chop regime turns ranging
            if (close[i] < h3_aligned[i] and close[i] > l3_aligned[i]) or \
               (close[i] < ema_50_1d_aligned[i]) or \
               (not chop_regime[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters Camarilla H3/L3 range OR 1d trend turns bullish OR chop regime turns ranging
            if (close[i] < h3_aligned[i] and close[i] > l3_aligned[i]) or \
               (close[i] > ema_50_1d_aligned[i]) or \
               (not chop_regime[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA50_Trend_VolumeConfirm_Regime"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 14:08

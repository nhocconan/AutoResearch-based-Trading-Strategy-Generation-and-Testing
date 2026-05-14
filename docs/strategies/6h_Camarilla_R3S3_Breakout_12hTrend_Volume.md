# Strategy: 6h_Camarilla_R3S3_Breakout_12hTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.272 | +33.9% | -9.0% | 142 | PASS |
| ETHUSDT | 0.264 | +35.3% | -16.6% | 133 | PASS |
| SOLUSDT | 0.931 | +137.4% | -17.7% | 94 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.106 | -6.0% | -10.6% | 52 | FAIL |
| ETHUSDT | 0.549 | +14.7% | -7.7% | 42 | PASS |
| SOLUSDT | -0.865 | -9.0% | -19.3% | 41 | FAIL |

## Code
```python
#!/usr/bin/env python3
# Hypothesis: 6h Camarilla R3/S3 breakout with 12h trend filter (HMA21) and volume confirmation (2.0x MA20).
# Enters long when price breaks above Camarilla R3 with 12h bullish trend and volume > 2.0x MA20.
# Enters short when price breaks below Camarilla S3 with 12h bearish trend and volume > 2.0x MA20.
# Exits when price crosses the 6h EMA20 (mean reversion).
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~12-37/year) by requiring strict confluence.
# Works in both bull and bear markets: 12h trend filter ensures alignment with higher timeframe direction,
# while Camarilla breakouts capture strong momentum moves and volume confirmation reduces false signals.
# Camarilla levels are derived from 1d OHLC (previous day) to avoid look-ahead.

name = "6h_Camarilla_R3S3_Breakout_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels (based on previous 1d bar)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R3, S3 (based on previous 1d bar)
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get 12h data for trend filter (HMA21)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    # Calculate HMA(21) on 12h close
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    wma_half = pd.Series(close_12h).ewm(span=half_len, adjust=False, min_periods=half_len).mean().values
    wma_full = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma_21 = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False, min_periods=sqrt_len).mean().values
    hma_21_aligned = align_htf_to_ltf(prices, df_12h, hma_21)
    
    # Volume filter: current volume > 2.0x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)
    
    # 6h EMA20 for exit condition
    ema20_6h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or \
           np.isnan(hma_21_aligned[i]) or np.isnan(vol_ma20[i]) or np.isnan(ema20_6h[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3 with 12h bullish trend and volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > hma_21_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3 with 12h bearish trend and volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < hma_21_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 6h EMA20 (mean reversion)
            if close[i] < ema20_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above 6h EMA20 (mean reversion)
            if close[i] > ema20_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-13 12:38

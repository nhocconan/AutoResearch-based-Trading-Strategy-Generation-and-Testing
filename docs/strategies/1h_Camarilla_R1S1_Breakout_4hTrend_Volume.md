# Strategy: 1h_Camarilla_R1S1_Breakout_4hTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.627 | -2.6% | -10.8% | 1040 | FAIL |
| ETHUSDT | 0.166 | +28.2% | -11.9% | 981 | PASS |
| SOLUSDT | 0.473 | +57.9% | -21.7% | 805 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.446 | +11.6% | -6.9% | 329 | PASS |
| SOLUSDT | 0.439 | +12.2% | -8.1% | 293 | PASS |

## Code
```python
#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R1/S1 breakout with 4h trend filter (EMA50) and volume spike (2x MA20).
# Enters long when price breaks above R1 level with 4h bullish trend (close > EMA50) and volume > 2.0x MA20.
# Enters short when price breaks below S1 level with 4h bearish trend (close < EMA50) and volume > 2.0x MA20.
# Exits when price crosses the 1h EMA20 (mean reversion).
# Uses discrete position sizing (0.20) to minimize fee drag.
# Designed for low trade frequency (~15-37/year) by requiring confluence of breakout, trend, and volume.
# Works in both bull and bear markets: trend filter ensures alignment with higher timeframe direction,
# while Camarilla levels provide precise intraday entry points with volume confirmation reducing false breakouts.

name = "1h_Camarilla_R1S1_Breakout_4hTrend_Volume"
timeframe = "1h"
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
    
    # Get 1h data for Camarilla pivot levels (based on previous 1h bar)
    df_1h = get_htf_data(prices, '1h')
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Calculate Camarilla levels: R1, S1 (based on previous 1h bar)
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    camarilla_r1 = close_1h + 1.1 * (high_1h - low_1h) / 12
    camarilla_s1 = close_1h - 1.1 * (high_1h - low_1h) / 12
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1h, camarilla_s1)
    
    # Get 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume filter: current volume > 2.0x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)
    
    # 1h EMA20 for exit condition
    ema20_1h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or \
           np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ma20[i]) or np.isnan(ema20_1h[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 with 4h bullish trend and volume spike
            if close[i] > camarilla_r1_aligned[i] and close[i] > ema50_4h_aligned[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S1 with 4h bearish trend and volume spike
            elif close[i] < camarilla_s1_aligned[i] and close[i] < ema50_4h_aligned[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 1h EMA20 (mean reversion)
            if close[i] < ema20_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price crosses above 1h EMA20 (mean reversion)
            if close[i] > ema20_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
```

## Last Updated
2026-05-13 12:31

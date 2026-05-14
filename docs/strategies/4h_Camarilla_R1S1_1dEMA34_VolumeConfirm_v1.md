# Strategy: 4h_Camarilla_R1S1_1dEMA34_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.323 | +29.7% | -10.3% | 321 | PASS |
| ETHUSDT | 0.269 | +29.6% | -6.1% | 294 | PASS |
| SOLUSDT | -0.283 | +6.6% | -14.9% | 260 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.656 | -3.3% | -5.6% | 129 | FAIL |
| ETHUSDT | 1.584 | +22.5% | -3.6% | 110 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Uses 4h timeframe (primary) and 1d HTF for EMA34 trend alignment (proven pattern from DB)
- Camarilla levels calculated from previous completed 4h bar's OHLC (based on prior 4h candle)
- Long when price breaks above Camarilla R1 AND price > 1d EMA34 (uptrend) AND volume > 2.0 * volume MA(20)
- Short when price breaks below Camarilla S1 AND price < 1d EMA34 (downtrend) AND volume > 2.0 * volume MA(20)
- Exit when price reverts to the Camarilla H3/L3 midpoint (mean reversion structure)
- Discrete signal size: 0.25 to minimize fee churn
- Target: 75-200 total trades over 4 years (19-50/year) as per 4h timeframe recommendation
- Works in both bull/bear: trend filter avoids counter-trend trades, Camarilla breakouts capture momentum in all regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Use previous completed 4h bar's OHLC for Camarilla calculation
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Shift by 1 to use previous completed 4h bar's OHLC
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_open = np.roll(prices['open'].values, 1)
    # First bar has no previous bar, set to NaN
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    prev_open[0] = np.nan
    
    # Calculate 1d EMA34 for trend filter (using previous completed 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough data for EMA34
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous completed 4h bar's OHLC
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    # Midpoint for exit: (H3 + L3)/2 = C (the close)
    range_val = prev_high - prev_low
    camarilla_R1 = prev_close + range_val * 1.1 / 12
    camarilla_S1 = prev_close - range_val * 1.1 / 12
    camarilla_H3 = prev_close + range_val * 1.1 / 4
    camarilla_L3 = prev_close - range_val * 1.1 / 4
    camarilla_mid = (camarilla_H3 + camarilla_L3) / 2.0  # equals prev_close
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * volume_ma)
    
    # Trend filter: price above/below 1d EMA34
    uptrend = close > ema_34_1d_aligned
    downtrend = close < ema_34_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need 1d EMA34, volume MA(20), and previous bar data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_R1[i]) or np.isnan(camarilla_S1[i]) or 
            np.isnan(camarilla_mid[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R1 AND uptrend AND volume confirmation
            if close[i] > camarilla_R1[i] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1 AND downtrend AND volume confirmation
            elif close[i] < camarilla_S1[i] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to Camarilla midpoint (previous close)
            if close[i] < camarilla_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to Camarilla midpoint (previous close)
            if close[i] > camarilla_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_1dEMA34_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-24 05:53

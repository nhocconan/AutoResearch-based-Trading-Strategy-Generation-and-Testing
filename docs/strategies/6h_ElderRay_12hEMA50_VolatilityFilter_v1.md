# Strategy: 6h_ElderRay_12hEMA50_VolatilityFilter_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.193 | +5.6% | -23.3% | 345 | FAIL |
| ETHUSDT | 0.105 | +23.6% | -15.8% | 359 | PASS |
| SOLUSDT | 0.883 | +178.1% | -24.2% | 350 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.085 | +6.2% | -12.2% | 118 | PASS |
| SOLUSDT | -0.022 | +3.4% | -11.8% | 119 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 12h EMA trend filter and ATR-based volatility filter.
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (using 13-period EMA on 6h)
- Long when Bull Power > 0 and rising AND price > 12h EMA50 (uptrend filter)
- Short when Bear Power < 0 and falling AND price < 12h EMA50 (downtrend filter)
- ATR filter: only trade when ATR(14) > 0.5 * ATR(50) to ensure sufficient volatility for meaningful moves
- Exit when Elder Ray power crosses zero or volatility drops below threshold
- Designed to capture institutional buying/selling pressure with trend alignment and volatility filter
- Signal size: 0.25 discrete levels to minimize fee churn
- Target: 50-150 total trades over 4 years (12-37/year)
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
    
    # Calculate 13-period EMA for Elder Ray (using 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Buying power: ability to push price above average
    bear_power = low - ema_13   # Selling power: ability to push price below average
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough data for EMA50
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Trend filter: price above/below 12h EMA50
    uptrend = close > ema_50_12h_aligned
    downtrend = close < ema_50_12h_aligned
    
    # ATR-based volatility filter: ATR(14) > 0.5 * ATR(50)
    # True Range calculation
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    volatility_filter = atr_14 > (0.5 * atr_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 50, 50)  # Need Elder Ray EMA, 12h EMA50, and ATR data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volatility_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 and rising AND uptrend AND sufficient volatility
            if bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and uptrend[i] and volatility_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 and falling AND downtrend AND sufficient volatility
            elif bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and downtrend[i] and volatility_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power crosses below zero OR volatility drops
            if bull_power[i] <= 0 or not volatility_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power crosses above zero OR volatility drops
            if bear_power[i] >= 0 or not volatility_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_12hEMA50_VolatilityFilter_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-24 05:24

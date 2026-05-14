# Strategy: 4h_12h_cci_volume_regime_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.284 | +34.9% | -9.2% | 250 | PASS |
| ETHUSDT | 0.363 | +42.7% | -11.4% | 227 | PASS |
| SOLUSDT | 1.103 | +185.0% | -20.1% | 217 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.032 | -4.3% | -7.8% | 79 | FAIL |
| ETHUSDT | 0.698 | +17.4% | -8.7% | 74 | PASS |
| SOLUSDT | 0.416 | +12.6% | -11.6% | 69 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_12h_cci_volume_regime_v2
# Strategy: 4h CCI with 12h trend filter and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: CCI (Commodity Channel Index) identifies cyclical trends and reversals.
# Long when CCI > 100 and rising, short when CCI < -100 and falling, with 12h EMA trend filter.
# Volume spike (1.5x 20-period average) confirms momentum strength. Designed for moderate
# frequency (25-40 trades/year) to balance signal quality and fee drag in bull/bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_cci_volume_regime_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # CCI calculation: (Typical Price - SMA) / (0.015 * Mean Deviation)
    period = 20
    tp = (high + low + close) / 3.0
    sma_tp = pd.Series(tp).rolling(window=period, min_periods=period).mean()
    mad = pd.Series(tp).rolling(window=period, min_periods=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    cci = (tp - sma_tp) / (0.015 * mad)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(cci.iloc[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # CCI signals: above/below threshold with momentum
        cci_now = cci.iloc[i]
        cci_prev = cci.iloc[i-1]
        cci_overbought = cci_now > 100 and cci_now > cci_prev
        cci_oversold = cci_now < -100 and cci_now < cci_prev
        
        # Entry logic: CCI extreme + volume spike + trend alignment
        if (cci_overbought and volume_spike[i] and uptrend and position != 1):
            position = 1
            signals[i] = 0.25
        elif (cci_oversold and volume_spike[i] and downtrend and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: CCI reversal or trend change
        elif position == 1 and (cci_now < 0 or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (cci_now > 0 or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
```

## Last Updated
2026-04-11 16:44

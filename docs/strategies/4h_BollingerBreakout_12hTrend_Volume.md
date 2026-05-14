# Strategy: 4h_BollingerBreakout_12hTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.409 | +31.9% | -5.7% | 272 | PASS |
| ETHUSDT | 0.370 | +32.0% | -3.3% | 261 | PASS |
| SOLUSDT | 0.021 | +20.4% | -18.0% | 255 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.999 | +1.1% | -3.3% | 103 | FAIL |
| ETHUSDT | 1.031 | +15.9% | -3.1% | 81 | PASS |
| SOLUSDT | -0.093 | +5.2% | -5.5% | 85 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Bands breakout with volume confirmation and 12h trend filter.
# Uses Bollinger Band breakouts (upper/lower bands) as entry signals, confirmed by volume spikes and 12h EMA trend.
# Designed to work in both bull and bear markets by following the 12h trend direction.
# Target: 20-50 trades/year per symbol to avoid excessive fee drag.
name = "4h_BollingerBreakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h trend filter: 50-period EMA on close
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Bollinger Bands (20, 2) on 4h close
    bb_period = 20
    bb_std = 2
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = sma_20 + bb_std * std_20
    bb_lower = sma_20 - bb_std * std_20
    
    # 4h volume average for spike detection
    vol_ema_4h = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = np.where(vol_ema_4h > 0, volume / vol_ema_4h, 1.0) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup for BB and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long breakout: price breaks above upper BB with volume spike in uptrend
            long_condition = (close[i] > bb_upper[i]) and vol_spike[i] and uptrend
            # Short breakdown: price breaks below lower BB with volume spike in downtrend
            short_condition = (close[i] < bb_lower[i]) and vol_spike[i] and downtrend
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price re-enters below upper BB or trend turns down
            if (close[i] < bb_upper[i]) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price re-enters above lower BB or trend turns up
            if (close[i] > bb_lower[i]) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-07 22:06

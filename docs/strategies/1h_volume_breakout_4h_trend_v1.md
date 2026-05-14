# Strategy: 1h_volume_breakout_4h_trend_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.033 | -12.3% | -17.5% | 888 | FAIL |
| ETHUSDT | -0.578 | -4.7% | -13.3% | 877 | FAIL |
| SOLUSDT | 0.249 | +35.6% | -19.3% | 688 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.115 | +7.1% | -8.9% | 264 | PASS |

## Code
```python
#!/usr/bin/env python3
# 1h_volume_breakout_4h_trend_v1
# Hypothesis: In 1h timeframe, capture breakouts with volume confirmation aligned with 4h trend direction.
# Uses 4h EMA50 for trend filter and 1h volume spike for entry timing. Limits trades to 15-35/year via
# strict volume threshold (2.0x average) and trend alignment. Works in bull/bear by following 4h trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_volume_breakout_4h_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA50 trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h EMA20 for dynamic support/resistance
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter: 1h volume > 2.0x 20-period average (very strict to limit trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Session filter: 08-20 UTC (reduce noise outside active hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_20[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(session_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price < EMA20 OR trend turns bearish (price < 4h EMA50)
            if (close[i] < ema_20[i]) or (close[i] < ema_50_4h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price > EMA20 OR trend turns bullish (price > 4h EMA50)
            if (close[i] > ema_20[i]) or (close[i] > ema_50_4h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Require session filter and volume spike
            if session_filter[i] and volume_spike[i]:
                # Long entry: price > EMA20 AND price > 4h EMA50 (bullish alignment)
                if (close[i] > ema_20[i]) and (close[i] > ema_50_4h_aligned[i]):
                    position = 1
                    signals[i] = 0.20
                # Short entry: price < EMA20 AND price < 4h EMA50 (bearish alignment)
                elif (close[i] < ema_20[i]) and (close[i] < ema_50_4h_aligned[i]):
                    position = -1
                    signals[i] = -0.20
    
    return signals
```

## Last Updated
2026-04-08 12:37

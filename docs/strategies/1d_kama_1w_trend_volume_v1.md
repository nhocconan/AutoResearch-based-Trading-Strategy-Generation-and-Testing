# Strategy: 1d_kama_1w_trend_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.835 | -3.8% | -15.5% | 47 | FAIL |
| ETHUSDT | -0.422 | +5.4% | -13.4% | 41 | FAIL |
| SOLUSDT | 0.672 | +77.8% | -12.1% | 45 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.257 | +9.2% | -7.3% | 15 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
1d_kama_1w_trend_volume_v1
Hypothesis: On 1d timeframe, use Kaufman Adaptive Moving Average (KAMA) as trend filter and breakout detector. Go long when price crosses above KAMA with volume confirmation and weekly uptrend (price > weekly EMA50). Go short when price crosses below KAMA with volume confirmation and weekly downtrend (price < weekly EMA50). KAMA adapts to market noise, reducing false signals in ranging markets while capturing trends. Weekly trend filter ensures alignment with higher timeframe momentum. Volume confirmation ensures signals have participation. Designed for low trade frequency (<25/year) to minimize fee impact.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    weekly_ema50 = df_weekly['close'].ewm(span=50, adjust=False).mean()
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema50.values)
    
    # Daily KAMA (10-period ER, 2/30 smoothing constants)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder
    # Proper volatility calculation: sum of absolute changes over ER period
    er_period = 10
    change_t = np.abs(np.diff(close, prepend=close[0]))
    volatility_sum = np.zeros_like(close)
    for i in range(er_period, len(close)):
        volatility_sum[i] = np.sum(change_t[i-er_period+1:i+1])
    net_change = np.abs(np.diff(close, prepend=close[0]))
    for i in range(er_period, len(close)):
        net_change[i] = np.abs(close[i] - close[i-er_period])
    er = np.zeros_like(close)
    er[er_period:] = np.where(volatility_sum[er_period:] > 0, net_change[er_period:] / volatility_sum[er_period:], 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # smoothing constant
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume confirmation (20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(weekly_ema50_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below KAMA or weekly trend turns bearish
            if close[i] < kama[i] or close[i] < weekly_ema50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses above KAMA or weekly trend turns bullish
            if close[i] > kama[i] or close[i] > weekly_ema50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price crosses above KAMA with volume in weekly uptrend
            if (close[i] > kama[i] and 
                vol_confirm and 
                close[i] > weekly_ema50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price crosses below KAMA with volume in weekly downtrend
            elif (close[i] < kama[i] and 
                  vol_confirm and 
                  close[i] < weekly_ema50_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-07 15:47

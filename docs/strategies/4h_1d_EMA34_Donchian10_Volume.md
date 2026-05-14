# Strategy: 4h_1d_EMA34_Donchian10_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.096 | +24.4% | -9.9% | 683 | PASS |
| ETHUSDT | 0.303 | +36.1% | -11.7% | 671 | PASS |
| SOLUSDT | 0.440 | +58.0% | -28.9% | 597 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.284 | -5.0% | -7.8% | 225 | FAIL |
| ETHUSDT | 0.329 | +10.7% | -6.1% | 239 | PASS |
| SOLUSDT | -0.626 | -3.9% | -15.6% | 196 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1-day trend filter (EMA34) and 4-hour Donchian breakout (10-period) with volume confirmation.
# Enters only during 08-20 UTC session. Uses strict conditions to limit trades (~15-25/year) and avoid overtrading.
# Trend-following in bull markets, avoids false signals in bear/chop via EMA34 filter and volume spike requirement.
name = "4h_1d_EMA34_Donchian10_Volume"
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
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA34 trend (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 4h data for Donchian10 breakout (called ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    # Donchian channels: 10-period high/low
    high_10_4h = pd.Series(high_4h).rolling(window=10, min_periods=10).max().values
    low_10_4h = pd.Series(low_4h).rolling(window=10, min_periods=10).min().values
    high_10_4h_aligned = align_htf_to_ltf(prices, df_4h, high_10_4h)
    low_10_4h_aligned = align_htf_to_ltf(prices, df_4h, low_10_4h)
    
    # Volume filter: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(high_10_4h_aligned[i]) or 
            np.isnan(low_10_4h_aligned[i]) or np.isnan(volume_ma[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above 1d EMA34 AND breaks 4h Donchian high with volume
            if (close[i] > ema_34_1d_aligned[i] and 
                close[i] > high_10_4h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below 1d EMA34 AND breaks 4h Donchian low with volume
            elif (close[i] < ema_34_1d_aligned[i] and 
                  close[i] < low_10_4h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below 1d EMA34 or 4h Donchian low
            if close[i] < ema_34_1d_aligned[i] or close[i] < low_10_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above 1d EMA34 or 4h Donchian high
            if close[i] > ema_34_1d_aligned[i] or close[i] > high_10_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-19 18:55

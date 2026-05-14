# Strategy: 6h_1d_EMA34_Donchian10_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.061 | +18.8% | -6.3% | 624 | FAIL |
| ETHUSDT | 0.076 | +23.5% | -7.5% | 641 | PASS |
| SOLUSDT | 0.150 | +27.7% | -15.2% | 512 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.191 | +8.0% | -9.4% | 212 | PASS |
| SOLUSDT | -0.639 | -1.7% | -7.3% | 209 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1-day trend filter (EMA34) and 6-hour Donchian breakout (10-period) with volume confirmation.
# Enters only during 08-20 UTC session. Uses strict conditions to limit trades (~15-25/year) and avoid overtrading.
# Trend-following in bull markets, avoids false signals in bear/chop via EMA34 filter and volume spike requirement.
name = "6h_1d_EMA34_Donchian10_Volume"
timeframe = "6h"
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
    
    # Get 6h data for Donchian10 breakout (called ONCE before loop)
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    # Donchian channels: 10-period high/low
    high_10_6h = pd.Series(high_6h).rolling(window=10, min_periods=10).max().values
    low_10_6h = pd.Series(low_6h).rolling(window=10, min_periods=10).min().values
    high_10_6h_aligned = align_htf_to_ltf(prices, df_6h, high_10_6h)
    low_10_6h_aligned = align_htf_to_ltf(prices, df_6h, low_10_6h)
    
    # Volume filter: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(high_10_6h_aligned[i]) or 
            np.isnan(low_10_6h_aligned[i]) or np.isnan(volume_ma[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above 1d EMA34 AND breaks 6h Donchian high with volume
            if (close[i] > ema_34_1d_aligned[i] and 
                close[i] > high_10_6h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below 1d EMA34 AND breaks 6h Donchian low with volume
            elif (close[i] < ema_34_1d_aligned[i] and 
                  close[i] < low_10_6h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below 1d EMA34 or 6h Donchian low
            if close[i] < ema_34_1d_aligned[i] or close[i] < low_10_6h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above 1d EMA34 or 6h Donchian high
            if close[i] > ema_34_1d_aligned[i] or close[i] > high_10_6h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-19 18:54

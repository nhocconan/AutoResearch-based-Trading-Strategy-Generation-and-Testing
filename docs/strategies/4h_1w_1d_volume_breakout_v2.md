# Strategy: 4h_1w_1d_volume_breakout_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.798 | +67.3% | -9.2% | 130 | PASS |
| ETHUSDT | 0.118 | +25.5% | -16.9% | 125 | PASS |
| SOLUSDT | 0.555 | +75.7% | -24.7% | 118 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.464 | +1.5% | -11.3% | 41 | FAIL |
| ETHUSDT | 0.999 | +21.5% | -17.3% | 39 | PASS |
| SOLUSDT | -0.053 | +4.0% | -14.5% | 48 | FAIL |

## Code
```python
# 4h_1w_1d_volume_breakout_v2
# Hypothesis: On 4h timeframe, price breaking above/below 1d high/low with volume expansion and weekly trend alignment captures breakout moves. Weekly trend filter avoids counter-trend breakouts in ranging markets. Volume confirmation filters false breakouts. Designed for both bull and bear markets.
# Entry: Long when price > 1d high + volume > 1.5x 20-period average + weekly uptrend
# Entry: Short when price < 1d low + volume > 1.5x 20-period average + weekly downtrend
# Exit: Opposite 1d level touch or weekly trend reversal
# Position sizing: 0.30 long, -0.30 short
# Uses 4h primary timeframe as required.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1w_1d_volume_breakout_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for daily high/low
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1w EMA(20) for trend
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_1w_up = close_1w > ema_20_1w
    trend_1w_down = close_1w < ema_20_1w
    
    # Forward fill trend
    trend_1w_up_series = pd.Series(trend_1w_up)
    trend_1w_down_series = pd.Series(trend_1w_down)
    trend_1w_up_ffilled = trend_1w_up_series.ffill().values
    trend_1w_down_ffilled = trend_1w_down_series.ffill().values
    
    # Align 1d high/low and 1w trend to 4h
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up_ffilled)
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down_ffilled)
    
    # Volume filter: 4h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(high_1d_aligned[i]) or np.isnan(low_1d_aligned[i]) or
            np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price touches 1d low OR weekly trend turns down
            if (close[i] <= low_1d_aligned[i]) or trend_1w_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30  # Position size
                
        elif position == -1:  # Short position
            # Exit: Price touches 1d high OR weekly trend turns up
            if (close[i] >= high_1d_aligned[i]) or trend_1w_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30  # Position size
        else:  # Flat, look for entry
            # Long entry: Price > 1d high + volume + weekly uptrend
            if (close[i] > high_1d_aligned[i]) and volume_filter[i] and trend_1w_up_aligned[i]:
                position = 1
                signals[i] = 0.30
            # Short entry: Price < 1d low + volume + weekly downtrend
            elif (close[i] < low_1d_aligned[i]) and volume_filter[i] and trend_1w_down_aligned[i]:
                position = -1
                signals[i] = -0.30
    
    return signals
```

## Last Updated
2026-04-08 12:15

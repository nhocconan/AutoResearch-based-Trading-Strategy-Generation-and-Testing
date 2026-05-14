# Strategy: 6h_1dPivot_R1S1_MomVol_Filter_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.242 | +27.2% | -2.6% | 162 | PASS |
| ETHUSDT | 0.147 | +25.2% | -4.7% | 141 | PASS |
| SOLUSDT | -0.491 | +2.2% | -18.1% | 126 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.299 | -4.4% | -4.8% | 63 | FAIL |
| ETHUSDT | 0.456 | +9.9% | -4.8% | 63 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using daily pivot points with volume confirmation and momentum filter
# Daily pivots provide key support/resistance. Breakouts above R1 or below S1 with volume > 2x 20-period average
# indicate strong momentum. Momentum filter (RSI > 55 for long, < 45 for short) ensures trend alignment.
# Works in bull/bear markets: breakouts capture trends, momentum filter avoids counter-trend trades.
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing.

name = "6h_1dPivot_R1S1_MomVol_Filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d pivot points ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for pivot calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Pivot point calculation
    pivot = (prev_high + prev_low + prev_close) / 3
    range_ = prev_high - prev_low
    
    # Support and Resistance levels
    r1 = pivot + range_
    s1 = pivot - range_
    
    # Align 1d levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: >2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_20)
    
    # Momentum filter: RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_long = rsi > 55
    rsi_short = rsi < 45
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(rsi_long[i]) or np.isnan(rsi_short[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume and momentum confirmation
            if close[i] > r1_aligned[i] and volume_filter[i] and rsi_long[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below S1 with volume and momentum confirmation
            elif close[i] < s1_aligned[i] and volume_filter[i] and rsi_short[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S1 (failed support) or RSI turns bearish
            if close[i] < s1_aligned[i] or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R1 (failed resistance) or RSI turns bullish
            if close[i] > r1_aligned[i] or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-06 22:01

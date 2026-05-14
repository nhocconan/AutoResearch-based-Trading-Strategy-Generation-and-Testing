# Strategy: 6h_12hPivot_R1S2_VolumeVolFilter_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.130 | +24.1% | -1.6% | 106 | PASS |
| ETHUSDT | 0.565 | +32.7% | -1.7% | 100 | PASS |
| SOLUSDT | -0.221 | +14.6% | -9.1% | 85 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.502 | -0.1% | -1.8% | 45 | FAIL |
| ETHUSDT | 0.300 | +8.0% | -2.5% | 43 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h pivot points with volume confirmation and volatility filter
# 12-hour pivots (R1/S1 for breakouts, R2/S2 for reversals) provide key intermediate levels
# Breakout above R1 or below S1 with volume > 2.0x 20-period average indicates strong momentum
# Rejection at R2 or S2 with volume confirmation indicates mean reversion within 12h range
# Volatility filter: ATR(14) > 20-period average ATR to avoid choppy markets
# Works in bull/bear markets: breakouts capture trends, reversals capture pullbacks within trend
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "6h_12hPivot_R1S2_VolumeVolFilter_v1"
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
    
    # Calculate 12h pivot points ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Previous 12h bar's OHLC for pivot calculation
    prev_close = df_12h['close'].shift(1).values
    prev_high = df_12h['high'].shift(1).values
    prev_low = df_12h['low'].shift(1).values
    
    # Pivot point calculation
    # Pivot = (previous high + previous low + previous close) / 3
    pivot = (prev_high + prev_low + prev_close) / 3
    range_ = prev_high - prev_low
    
    # Support and Resistance levels
    r1 = pivot + (range_ * 1.0)
    r2 = pivot + (range_ * 2.0)
    s1 = pivot - (range_ * 1.0)
    s2 = pivot - (range_ * 2.0)
    
    # Align 12h levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    r2_aligned = align_htf_to_ltf(prices, df_12h, r2)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    s2_aligned = align_htf_to_ltf(prices, df_12h, s2)
    
    # Volume confirmation: >2.0x 20-period average (higher threshold to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_20)
    
    # Volatility filter: ATR(14) > 20-period average ATR
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(np.maximum(tr1, tr2), tr3)])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    vol_filter = atr_14 > atr_ma_20
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(volume_filter[i]) or np.isnan(vol_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume confirmation and volatility
            if close[i] > r1_aligned[i] and volume_filter[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below S1 with volume confirmation and volatility
            elif close[i] < s1_aligned[i] and volume_filter[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
            # Long reversal: price rejects S2 with volume confirmation and volatility
            elif close[i] < s2_aligned[i] and close[i] > s2_aligned[i] * 0.995 and volume_filter[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short reversal: price rejects R2 with volume confirmation and volatility
            elif close[i] > r2_aligned[i] and close[i] < r2_aligned[i] * 1.005 and volume_filter[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S1 (failed support) or reaches R2 (take profit)
            if close[i] < s1_aligned[i] or close[i] > r2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R1 (failed resistance) or reaches S2 (take profit)
            if close[i] > r1_aligned[i] or close[i] < s2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-06 21:56

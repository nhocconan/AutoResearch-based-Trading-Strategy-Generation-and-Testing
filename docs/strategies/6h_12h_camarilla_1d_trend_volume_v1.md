# Strategy: 6h_12h_camarilla_1d_trend_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.318 | +30.4% | -3.7% | 307 | PASS |
| ETHUSDT | -0.232 | +13.7% | -6.4% | 290 | FAIL |
| SOLUSDT | 0.237 | +33.7% | -17.4% | 254 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.377 | -9.1% | -9.3% | 131 | FAIL |
| SOLUSDT | 0.429 | +10.1% | -9.1% | 80 | PASS |

## Code
```python
#!/usr/bin/env python3
# 6h_12h_camarilla_1d_trend_volume_v1
# Hypothesis: Camarilla pivot levels from 12h with 1d EMA trend filter and volume confirmation.
# Uses 12h Camarilla levels (R3/S3 for fade, R4/S4 for breakout) and 1d EMA to filter trend direction.
# Volume confirmation ensures institutional participation. Designed for 6h timeframe to capture
# multi-day swings while avoiding excessive trading. Target: 15-30 trades/year.

name = "6h_12h_camarilla_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h Camarilla pivot levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 12h bar
    # Using typical formula: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4)
    # S3 = C - ((H-L) * 1.1/4), S4 = C - ((H-L) * 1.1/2)
    # where C = (H+L+C)/3 (typical price)
    typical_price = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    hl_range = df_12h['high'] - df_12h['low']
    
    r4 = typical_price + (hl_range * 1.1 / 2)
    r3 = typical_price + (hl_range * 1.1 / 4)
    s3 = typical_price - (hl_range * 1.1 / 4)
    s4 = typical_price - (hl_range * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4.values)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3.values)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4.values)
    
    # 1d EMA trend filter (34-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: volume > 1.8x 24-period average (~4 days on 6h)
    vol_period = 24
    vol_ma = np.full(n, np.nan)
    if n >= vol_period:
        vol_ma[vol_period-1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period-1:].values
    
    # Start from sufficient lookback
    start_idx = max(24, 1) + 5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 1.8 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below R3 or trend fails
            if close[i] < r3_aligned[i] or close[i] < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above S3 or trend fails
            if close[i] > s3_aligned[i] or close[i] > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade with volume confirmation
            if volume_filter:
                # Fade at S3/R3: price rejects at support/resistance with trend
                # Long: bounce from S3 with uptrend
                if close[i] > s3_aligned[i] and close[i] > ema_1d_aligned[i] and low[i] <= s3_aligned[i] * 1.001:
                    position = 1
                    signals[i] = 0.25
                # Short: rejection at R3 with downtrend
                elif close[i] < r3_aligned[i] and close[i] < ema_1d_aligned[i] and high[i] >= r3_aligned[i] * 0.999:
                    position = -1
                    signals[i] = -0.25
                # Breakout at S4/R4: strong momentum continuation
                # Long: break above R4 with uptrend
                elif close[i] > r4_aligned[i] and close[i] > ema_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: break below S4 with downtrend
                elif close[i] < s4_aligned[i] and close[i] < ema_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-08 08:26

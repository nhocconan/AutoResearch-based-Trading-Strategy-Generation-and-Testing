# Strategy: 4H_Camarilla_R3S3_Breakout_1dEMA100_Trend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.290 | +5.0% | -14.8% | 43 | FAIL |
| ETHUSDT | 0.029 | +20.2% | -16.0% | 38 | PASS |
| SOLUSDT | 0.950 | +134.2% | -23.6% | 31 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.102 | +25.8% | -7.3% | 14 | PASS |
| SOLUSDT | -1.022 | -12.5% | -27.4% | 12 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4H_Camarilla_R3S3_Breakout_1dEMA100_Trend_Volume
Hypothesis: Daily EMA100 defines long-term trend, daily Camarilla R3/S3 levels act as strong support/resistance.
In bull markets, buy breakouts above R3 with daily uptrend. In bear markets, sell breakdowns below S3 with daily downtrend.
Volume spike confirms institutional interest. Uses 4h timeframe for better trade frequency and lower fee drag vs 1d.
Target: 20-50 trades/year, low turnover to minimize fee drag in ranging 2025 market.
"""

name = "4H_Camarilla_R3S3_Breakout_1dEMA100_Trend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema100_1d = pd.Series(close_1d).ewm(span=100, min_periods=100, adjust=False).mean().values
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # Load daily data ONCE for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R3, S3 (outer levels for fewer, stronger signals)
    hl_range = high_1d - low_1d
    r3 = close_1d + hl_range * 1.5000
    s3 = close_1d - hl_range * 1.5000
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume filter: 20-period EMA for spike detection (using 4h volume)
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # Fixed position size to minimize churn
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema100_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_ema1d = close[i] > ema100_1d_aligned[i]
        price_below_ema1d = close[i] < ema100_1d_aligned[i]
        breakout_long = close[i] > r3_aligned[i]
        breakout_short = close[i] < s3_aligned[i]
        
        if position == 0:
            # Long: Price breaks above R3 + above daily EMA100 + volume spike
            if breakout_long and price_above_ema1d and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Price breaks below S3 + below daily EMA100 + volume spike
            elif breakout_short and price_below_ema1d and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions - simplified to reduce churn
            if position == 1:
                # Exit: Price crosses below S3 OR trend reverses (close below daily EMA)
                if close[i] < s3_aligned[i] or close[i] < ema100_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price crosses above R3 OR trend reverses (close above daily EMA)
                if close[i] > r3_aligned[i] or close[i] > ema100_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals
```

## Last Updated
2026-05-11 10:06

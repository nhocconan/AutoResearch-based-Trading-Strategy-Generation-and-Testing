# Strategy: 6h_Institutional_Flow_Detector_1dTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.300 | +31.5% | -7.8% | 182 | PASS |
| ETHUSDT | 0.433 | +39.6% | -10.6% | 151 | PASS |
| SOLUSDT | 0.472 | +55.9% | -19.6% | 133 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.429 | +2.5% | -3.7% | 72 | FAIL |
| ETHUSDT | 2.037 | +34.7% | -5.7% | 65 | PASS |
| SOLUSDT | -0.142 | +3.9% | -6.6% | 53 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h Institutional Flow Detector with 1d Trend Filter and Volume Confirmation.
Long when: 1) Institutional flow > 0 (large volume + price close near high), 2) Price > 1d EMA50, 3) Volume > 2x average.
Short when: 1) Institutional flow < 0 (large volume + price close near low), 2) Price < 1d EMA50, 3) Volume > 2x average.
Exit when institutional flow crosses zero.
Uses 1d EMA50 for trend filter to avoid counter-trend trades.
Designed for 6h timeframe: targets 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Institutional Flow: (close - low) - (high - close) normalized by range, scaled by volume
    # Positive = buying pressure (close near high), Negative = selling pressure (close near low)
    price_range = high - low
    # Avoid division by zero
    price_range = np.where(price_range == 0, 1, price_range)
    money_flow_multiplier = ((close - low) - (high - close)) / price_range  # [-1, 1]
    money_flow_volume = money_flow_multiplier * volume
    
    # Sum over 6 periods (1 day equivalent on 6h chart) for institutional flow
    inst_flow = np.full(n, np.nan, dtype=np.float64)
    for i in range(5, n):
        inst_flow[i] = np.sum(money_flow_volume[i-5:i+1])
    
    # Volume filter: volume > 2x 24-period average (48 hours = 2 days)
    vol_ma_24 = np.full(n, np.nan, dtype=np.float64)
    for i in range(23, n):
        vol_ma_24[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need institutional flow (5 periods) + volume MA + 1d EMA
    start_idx = max(23, 50, 5)  # Need all indicators
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(inst_flow[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        flow_now = inst_flow[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_now = volume[i]
        vol_avg = vol_ma_24[i]
        
        # Volume filter: volume > 2x average
        vol_filter = vol_now > 2.0 * vol_avg
        
        if position == 0:
            # Long: institutional buying + price > 1d EMA50 + volume spike
            if flow_now > 0 and close[i] > ema_trend and vol_filter:
                signals[i] = size
                position = 1
            # Short: institutional selling + price < 1d EMA50 + volume spike
            elif flow_now < 0 and close[i] < ema_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: institutional flow turns negative (selling pressure)
            if flow_now < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: institutional flow turns positive (buying pressure)
            if flow_now > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Institutional_Flow_Detector_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-27 09:03

# Strategy: 6h_1d_rvol_breakout_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.432 | +49.0% | -12.1% | 46 | PASS |
| ETHUSDT | 0.162 | +28.6% | -15.4% | 41 | PASS |
| SOLUSDT | 0.744 | +138.0% | -29.8% | 40 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.737 | -3.9% | -11.8% | 21 | FAIL |
| ETHUSDT | 1.044 | +29.4% | -8.9% | 16 | PASS |
| SOLUSDT | 0.130 | +7.1% | -24.0% | 11 | PASS |

## Code
```python
#!/usr/bin/env python3
# 6h_1d_rvol_breakout_v1
# Strategy: 6-hour Relative Volume (RVOL) breakout with 1-day trend filter
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Price breakouts above/below 6h Donchian channels (20-period) combined with
# RVOL > 2.0 (current volume > 2x 20-period average volume) capture institutional
# momentum. The 1-day EMA(50) trend filter ensures trades align with higher timeframe
# direction, reducing false breakouts in sideways markets. Works in bull by catching
# continuation breakouts and in bear by capturing breakdowns with volume confirmation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_rvol_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h Donchian channel (20-period) for breakout
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 6h Relative Volume (RVOL): current volume / 20-period average volume
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    rvol = volume / (vol_avg_20 + 1e-10)  # Avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(rvol[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions
        bull_breakout = close[i] > donchian_high[i-1]  # Break above prior high
        bear_breakout = close[i] < donchian_low[i-1]   # Break below prior low
        
        # Volume confirmation: RVOL > 2.0
        vol_confirm = rvol[i] > 2.0
        
        # Trend filter: price above/below daily EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry logic: breakout + volume + trend alignment
        if bull_breakout and vol_confirm and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif bear_breakout and vol_confirm and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite breakout with volume confirmation
        elif position == 1 and bear_breakout and vol_confirm:
            position = 0
            signals[i] = 0.0
        elif position == -1 and bull_breakout and vol_confirm:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
```

## Last Updated
2026-04-11 13:58

# Strategy: 6h_Bollinger_Breakout_1dEMA50_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.250 | +31.4% | -9.2% | 108 | PASS |
| ETHUSDT | 0.084 | +23.7% | -11.8% | 97 | PASS |
| SOLUSDT | 0.453 | +60.1% | -28.7% | 98 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.118 | -4.3% | -7.2% | 39 | FAIL |
| ETHUSDT | 0.922 | +21.0% | -5.6% | 34 | PASS |
| SOLUSDT | -0.240 | +1.3% | -14.9% | 30 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band breakout with 1d trend filter and volume confirmation
# Uses Bollinger Bands (20,2.0) for volatility-based breakout detection
# 1d EMA50 filter ensures trades align with higher timeframe trend
# Volume spike (1.8x 24-bar MA) confirms institutional participation
# Designed for 50-150 total trades over 4 years (12-37/year) on 6h timeframe
# Works in bull markets (breakouts with trend) and bear markets (mean reversion at band extremes)

name = "6h_Bollinger_Breakout_1dEMA50_Volume"
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
    
    # Bollinger Bands (20, 2.0) on 6h timeframe
    close_s = pd.Series(close)
    bb_ma = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_ma + 2.0 * bb_std
    bb_lower = bb_ma - 2.0 * bb_std
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 1.8x 24-period average (24*6h = 6 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Bollinger Bands and volume MA)
    start_idx = 24
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above upper band AND price > 1d EMA50 (bullish trend) AND volume spike
            if (close[i] > bb_upper[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below lower band AND price < 1d EMA50 (bearish trend) AND volume spike
            elif (close[i] < bb_lower[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below middle band OR price below 1d EMA50 (trend change)
            if close[i] < bb_ma[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above middle band OR price above 1d EMA50 (trend change)
            if close[i] > bb_ma[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-02 08:08

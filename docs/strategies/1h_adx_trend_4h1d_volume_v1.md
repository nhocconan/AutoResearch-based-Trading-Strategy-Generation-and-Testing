# Strategy: 1h_adx_trend_4h1d_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.174 | +28.2% | -9.9% | 843 | PASS |
| ETHUSDT | 0.223 | +32.2% | -13.2% | 828 | PASS |
| SOLUSDT | 0.621 | +86.1% | -25.6% | 773 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.067 | -3.5% | -7.3% | 270 | FAIL |
| ETHUSDT | 0.364 | +10.9% | -8.8% | 275 | PASS |
| SOLUSDT | -0.080 | +3.8% | -10.4% | 245 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h ADX Trend + 4h/1d Higher Timeframe Trend + Volume Spike
# Hypothesis: Strong trends on 4h/1d with volume confirmation on 1h capture
# institutional moves while avoiding counter-trend noise. ADX filters weak trends.
# Target: 15-37 trades/year (60-150 total over 4 years) for 1h timeframe.

name = "1h_adx_trend_4h1d_volume_v1"
timeframe = "1h"
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 4h EMA(20) for trend filter
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1d EMA(20) for trend filter
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # ADX(14) on 1h to measure trend strength
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr_ma = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    dm_plus_ma = pd.Series(dm_plus).ewm(span=14, adjust=False).mean().values
    dm_minus_ma = pd.Series(dm_minus).ewm(span=14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_ma / tr_ma
    di_minus = 100 * dm_minus_ma / tr_ma
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(ema_20_1d_aligned[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: trend turns bearish or ADX weak
            if close[i] < ema_20_4h_aligned[i] or close[i] < ema_20_1d_aligned[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short position
            # Exit: trend turns bullish or ADX weak
            if close[i] > ema_20_4h_aligned[i] or close[i] > ema_20_1d_aligned[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            if vol_ok and adx[i] > 25:  # Strong trend + volume
                # Strong uptrend on both timeframes
                if close[i] > ema_20_4h_aligned[i] and close[i] > ema_20_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Strong downtrend on both timeframes
                elif close[i] < ema_20_4h_aligned[i] and close[i] < ema_20_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals
```

## Last Updated
2026-04-07 13:12

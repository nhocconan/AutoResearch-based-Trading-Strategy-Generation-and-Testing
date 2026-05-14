# Strategy: 4h_12hEMA34Filter_PivotBreakout_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.159 | +27.0% | -11.6% | 207 | PASS |
| ETHUSDT | 0.066 | +22.7% | -11.0% | 200 | PASS |
| SOLUSDT | 0.864 | +116.6% | -18.3% | 186 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.798 | -7.9% | -8.0% | 75 | FAIL |
| ETHUSDT | 0.367 | +10.6% | -9.6% | 71 | PASS |
| SOLUSDT | -0.877 | -6.3% | -11.3% | 63 | FAIL |

## Code
```python
#!/usr/bin/env python3
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
    
    # Get 12h data for trend filter (per experiment instruction)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(34) for trend filter
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h EMA to 4h timeframe
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Get 1d data for pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    
    # Use previous day's pivots (avoid look-ahead)
    r1_prev = np.roll(r1, 1)
    s1_prev = np.roll(s1, 1)
    r2_prev = np.roll(r2, 1)
    s2_prev = np.roll(s2, 1)
    r1_prev[0] = np.nan
    s1_prev[0] = np.nan
    r2_prev[0] = np.nan
    s2_prev[0] = np.nan
    
    # Align daily pivot levels to 4h timeframe
    r2_4h = align_htf_to_ltf(prices, df_1d, r2_prev)
    s2_4h = align_htf_to_ltf(prices, df_1d, s2_prev)
    
    # Volume confirmation: current volume > 1.5 * 20-period average (approx 5 days)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR filter to avoid low volatility environments
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma10 = pd.Series(atr).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 40  # Need EMA34, R2/S2, volume MA20, ATR MA10
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma20[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(atr_ma10[i]) or 
            np.isnan(r2_4h[i]) or 
            np.isnan(s2_4h[i]) or
            np.isnan(ema34_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        # Volatility filter: ATR > ATR MA10 (avoid low volatility)
        volatility_filter = atr[i] > atr_ma10[i]
        # 12h trend filter: price above/below 12h EMA34
        trend_up = close[i] > ema34_12h_aligned[i]
        trend_down = close[i] < ema34_12h_aligned[i]
        
        if position == 0:
            # Long: price breaks above R2 with volume, volatility AND 12h uptrend
            if close[i] > r2_4h[i] and volume_filter and volatility_filter and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S2 with volume, volatility AND 12h downtrend
            elif close[i] < s2_4h[i] and volume_filter and volatility_filter and trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below 12h EMA34 or volatility drops
            if close[i] < ema34_12h_aligned[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above 12h EMA34 or volatility drops
            if close[i] > ema34_12h_aligned[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12hEMA34Filter_PivotBreakout_Volume"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-17 09:19

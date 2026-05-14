# Strategy: 6h_Supertrend_1dTrend_VolumeFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.094 | +13.2% | -13.1% | 94 | DISCARD |
| ETHUSDT | 0.032 | +18.9% | -13.5% | 84 | KEEP |
| SOLUSDT | 0.493 | +76.2% | -34.1% | 87 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.353 | +12.0% | -11.1% | 28 | KEEP |
| SOLUSDT | 0.064 | +5.7% | -12.7% | 29 | KEEP |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Supertrend_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Supertrend on 6h data (ATR=10, multiplier=3)
    atr_period = 10
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Basic upper and lower bands
    hl2 = (high + low) / 2
    upper_band = hl2 + 3 * atr
    lower_band = hl2 - 3 * atr
    
    # Initialize Supertrend
    supertrend = np.full(n, np.nan)
    direction = np.full(n, 1)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, n):
        if close[i] > upper_band[i-1]:
            direction[i] = 1
        elif close[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Align Supertrend to 6h (already on 6h, but ensure alignment)
    supertrend_6h = supertrend  # already aligned to 6h
    
    # Volume filter: above 1.5x 24-period average (24*6h = 144h ~ 6 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(supertrend_6h[i]) or np.isnan(ema_50_6h[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma[i]  # Volume confirmation
        
        # Pre-compute hour for session filter (UTC 8-20)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long: Supertrend uptrend + price above EMA50 + volume + session
            if (supertrend_6h[i] < close[i] and  # price above Supertrend (uptrend)
                close[i] > ema_50_6h[i] and     # 1d uptrend filter
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short: Supertrend downtrend + price below EMA50 + volume + session
            elif (supertrend_6h[i] > close[i] and  # price below Supertrend (downtrend)
                  close[i] < ema_50_6h[i] and      # 1d downtrend filter
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Supertrend (trend change)
            if supertrend_6h[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Supertrend (trend change)
            if supertrend_6h[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-09 12:03

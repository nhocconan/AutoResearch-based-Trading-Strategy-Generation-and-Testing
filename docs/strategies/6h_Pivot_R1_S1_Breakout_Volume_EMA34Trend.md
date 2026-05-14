# Strategy: 6h_Pivot_R1_S1_Breakout_Volume_EMA34Trend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.225 | +28.3% | -6.3% | 126 | PASS |
| ETHUSDT | 0.344 | +34.6% | -7.2% | 113 | PASS |
| SOLUSDT | 0.090 | +23.8% | -17.5% | 82 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.379 | -6.6% | -7.3% | 44 | FAIL |
| ETHUSDT | 0.707 | +13.7% | -4.9% | 39 | PASS |
| SOLUSDT | 0.072 | +6.6% | -7.0% | 31 | PASS |

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
    
    # === Daily data for pivot points and volatility ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points: P, R1, S1
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    r1 = pivot + range_hl
    s1 = pivot - range_hl
    
    # Calculate daily ATR (14-period) for volatility filter
    tr_1d = np.maximum(high_1d - low_1d,
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Align daily data to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    atr_6h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 6h EMA34 for trend filter (primary timeframe)
    ema_6h = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume spike detection (20-period volume MA on 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(atr_6h[i]) or np.isnan(ema_6h[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        pivot_level = pivot_6h[i]
        r1_level = r1_6h[i]
        s1_level = s1_6h[i]
        atr = atr_6h[i]
        ema_trend = ema_6h[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price returns to pivot level (mean reversion to daily pivot)
            if price <= pivot_level:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price returns to pivot level (mean reversion to daily pivot)
            if price >= pivot_level:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R1 with volume spike, volatility filter, and uptrend
            if price > r1_level and vol_spike and atr > 0 and price > ema_trend:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below S1 with volume spike, volatility filter, and downtrend
            elif price < s1_level and vol_spike and atr > 0 and price < ema_trend:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Pivot_R1_S1_Breakout_Volume_EMA34Trend"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-16 16:07

# Strategy: 6h_1d_pivot_ema50_vol_filter_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.391 | -0.1% | -13.8% | 122 | FAIL |
| ETHUSDT | 0.001 | +17.4% | -17.8% | 118 | PASS |
| SOLUSDT | 0.674 | +104.5% | -25.7% | 84 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.993 | +25.8% | -7.7% | 36 | PASS |
| SOLUSDT | 0.865 | +24.2% | -8.8% | 30 | PASS |

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
    
    # Get daily data for 12h EMA and pivot calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 50-period EMA on daily for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate daily pivots (standard floor pivot)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    r2_1d = pivot_1d + (high_1d - low_1d)
    s2_1d = pivot_1d - (high_1d - low_1d)
    r3_1d = high_1d + 2 * (pivot_1d - low_1d)
    s3_1d = low_1d - 2 * (high_1d - pivot_1d)
    
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Calculate 6-period ATR for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr6 = np.full(n, np.nan)
    for i in range(5, n):
        atr6[i] = np.nanmean(tr[i-5:i+1])
    
    # Calculate 6-period ATR EMA for volatility regime
    atr_series = pd.Series(atr6)
    atr_ema6 = atr_series.ewm(span=6, adjust=False, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(atr6[i]) or np.isnan(atr_ema6[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current ATR6 > 1.0x 6-period ATR EMA (moderate volatility)
        vol_filter = atr6[i] > atr_ema6[i] * 1.0
        
        # Trend filter: price above/below daily 50 EMA
        price_above_ema50 = close[i] > ema50_1d_aligned[i]
        price_below_ema50 = close[i] < ema50_1d_aligned[i]
        
        # Entry conditions: bounce from S1/S2 with trend alignment
        long_bounce_s1 = (low[i] <= s1_aligned[i] * 1.002) and (close[i] > s1_aligned[i])
        long_bounce_s2 = (low[i] <= s2_aligned[i] * 1.002) and (close[i] > s2_aligned[i])
        long_entry = (long_bounce_s1 or long_bounce_s2) and price_above_ema50 and vol_filter
        
        # Entry conditions: rejection at R1/R2 with trend alignment
        short_reject_r1 = (high[i] >= r1_aligned[i] * 0.998) and (close[i] < r1_aligned[i])
        short_reject_r2 = (high[i] >= r2_aligned[i] * 0.998) and (close[i] < r2_aligned[i])
        short_entry = (short_reject_r1 or short_reject_r2) and price_below_ema50 and vol_filter
        
        # Exit conditions: opposite signal or volatility drop
        long_exit = (close[i] < ema50_1d_aligned[i]) or (atr6[i] < atr_ema6[i] * 0.7)
        short_exit = (close[i] > ema50_1d_aligned[i]) or (atr6[i] < atr_ema6[i] * 0.7)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_pivot_ema50_vol_filter_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-12 18:47

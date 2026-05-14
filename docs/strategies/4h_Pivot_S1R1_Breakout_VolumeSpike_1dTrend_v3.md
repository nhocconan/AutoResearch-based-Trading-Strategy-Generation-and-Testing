# Strategy: 4h_Pivot_S1R1_Breakout_VolumeSpike_1dTrend_v3

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.320 | +34.6% | -11.8% | 93 | PASS |
| ETHUSDT | 0.238 | +32.3% | -9.8% | 85 | PASS |
| SOLUSDT | 1.156 | +162.5% | -15.3% | 66 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.936 | -1.8% | -7.3% | 30 | FAIL |
| ETHUSDT | 0.103 | +6.9% | -10.7% | 33 | PASS |
| SOLUSDT | -0.302 | +0.8% | -15.4% | 20 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h_Pivot_S1R1_Breakout_VolumeSpike_1dTrend_v3
Hypothesis: Daily pivot S1/R1 levels act as strong support/resistance. Breakouts with volume spike and daily EMA(34) trend filter capture momentum. Added tighter volume requirement and minimum holding period to reduce trade frequency and improve quality. Target: 15-25 trades/year on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculation and EMA (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points using standard formula
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    close_1d = df_1d['close']
    
    pivot = (high_1d + low_1d + close_1d) / 3
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Shift by 1 to use previous day's levels only
    r1_prev = r1.shift(1).values
    s1_prev = s1.shift(1).values
    
    # Align to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_prev)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_prev)
    
    # Get daily data for EMA trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # ATR for volatility filter (14-period)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volatility filter: only trade when ATR > 20-period average (avoid chop)
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    volatility_filter = atr > atr_ma
    
    # Volume spike: 2.5x 20-period average on 4h (tighter than before)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    bars_since_entry = 0  # track holding period
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i]) or
            np.isnan(atr_ma[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_filter = volatility_filter[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            bars_since_entry = 0
            # Long: break above R1 with volume spike, price above daily EMA, and sufficient volatility
            if price > r1_val and vol_spike and price > ema_trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume spike, price below daily EMA, and sufficient volatility
            elif price < s1_val and vol_spike and price < ema_trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Minimum holding period: 4 bars (16 hours for 4h)
            if bars_since_entry < 4:
                signals[i] = 0.25
                bars_since_entry += 1
            else:
                signals[i] = 0.25
                # Exit: price returns to S1 or breaks below daily EMA
                if price <= s1_val or price < ema_trend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
        
        elif position == -1:
            # Minimum holding period: 4 bars (16 hours for 4h)
            if bars_since_entry < 4:
                signals[i] = -0.25
                bars_since_entry += 1
            else:
                signals[i] = -0.25
                # Exit: price returns to R1 or breaks above daily EMA
                if price >= r1_val or price > ema_trend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
    
    return signals

name = "4h_Pivot_S1R1_Breakout_VolumeSpike_1dTrend_v3"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-18 01:47

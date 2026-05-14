# Strategy: 4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_12hTrend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.422 | +36.7% | -8.1% | 344 | PASS |
| ETHUSDT | 0.196 | +28.8% | -8.3% | 311 | PASS |
| SOLUSDT | 0.124 | +25.8% | -14.6% | 262 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.645 | -5.7% | -6.7% | 139 | FAIL |
| ETHUSDT | 1.335 | +23.4% | -5.9% | 116 | PASS |
| SOLUSDT | 1.372 | +23.9% | -4.7% | 97 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_12hTrend
Hypothesis: On 4h timeframe, Camarilla R3/S3 breakouts with 1d EMA34 trend filter and volume spike (>2.0x 20-bar avg) capture high-probability institutional breakouts. Added 12h EMA50 trend filter to ensure alignment with intermediate-term trend, reducing false signals in choppy markets. Discrete sizing (0.25) minimizes fee churn. Works in bull/bear via trend filter adaptation.
"""

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
    
    # Get 1d data for HTF trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on 1d for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 12h data for intermediate trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 on 12h for intermediate trend
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 4h data for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels from previous 4h bar (R3, S3)
    prev_close = np.concatenate([[np.nan], close_4h[:-1]])
    prev_high = np.concatenate([[np.nan], high_4h[:-1]])
    prev_low = np.concatenate([[np.nan], low_4h[:-1]])
    
    camarilla_range = prev_high - prev_low
    r3 = prev_close + 1.1 * camarilla_range * 1.1 / 4
    s3 = prev_close - 1.1 * camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3)
    
    # Volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(50, 20, 34, 50)  # EMA34, vol MA, EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_34_val = ema_34_aligned[i]
        ema_50_12h_val = ema_50_12h_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current volume > 2.0x 20-period average
        volume_spike = vol_val > 2.0 * vol_ma_val
        
        if position == 0:
            # Long: price breaks above R3 with uptrend (close > EMA34 and EMA50_12h) and volume spike
            long_signal = (high_val > r3_val) and (close_val > ema_34_val) and (close_val > ema_50_12h_val) and volume_spike
            # Short: price breaks below S3 with downtrend (close < EMA34 and EMA50_12h) and volume spike
            short_signal = (low_val < s3_val) and (close_val < ema_34_val) and (close_val < ema_50_12h_val) and volume_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks below S3 (exit long)
            if close_val < s3_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks above R3 (exit short)
            if close_val > r3_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_12hTrend"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-26 00:07

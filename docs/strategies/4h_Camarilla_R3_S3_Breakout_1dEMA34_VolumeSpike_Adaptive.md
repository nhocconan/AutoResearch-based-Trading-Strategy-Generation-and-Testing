# Strategy: 4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike_Adaptive

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.188 | +29.0% | -9.0% | 2069 | KEEP |
| ETHUSDT | 0.244 | +33.4% | -12.5% | 2129 | KEEP |
| SOLUSDT | 0.987 | +157.3% | -18.7% | 1896 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.486 | +1.7% | -7.1% | 482 | DISCARD |
| ETHUSDT | 1.161 | +26.3% | -6.9% | 686 | KEEP |
| SOLUSDT | -0.149 | +2.5% | -10.3% | 497 | DISCARD |

## Code
```python
#!/usr/bin/env python3
name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike_Adaptive"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate daily high/low/close for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R3, R2, R1, PP, S1, S2, S3
    hl_range = high_1d - low_1d
    r3 = close_1d + hl_range * 1.25
    s3 = close_1d - hl_range * 1.25
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume filter: 50-period EMA for higher threshold
    vol_ema50 = pd.Series(volume).ewm(span=50, min_periods=50, adjust=False).mean().values
    volume_ok = volume > vol_ema50 * 2.5  # Further increased threshold to reduce trades
    
    # Adaptive position sizing based on volatility
    atr_period = 14
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.insert(tr, 0, high[0] - low[0])
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    atr_norm = atr / close
    vol_percentile = pd.Series(atr_norm).rolling(window=100, min_periods=100).rank(pct=True).values
    # Scale position size: 0.15 in low vol, 0.30 in high vol
    position_size = 0.15 + 0.15 * vol_percentile
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_ok[i]) or np.isnan(position_size[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_ema1d = close[i] > ema34_1d_aligned[i]
        price_below_ema1d = close[i] < ema34_1d_aligned[i]
        breakout_long = close[i] > r3_aligned[i]
        breakout_short = close[i] < s3_aligned[i]
        
        if position == 0:
            # Long: Price breaks above R3 + above 1d EMA34 + volume spike
            if breakout_long and price_above_ema1d and volume_ok[i]:
                signals[i] = position_size[i]
                position = 1
            # Short: Price breaks below S3 + below 1d EMA34 + volume spike
            elif breakout_short and price_below_ema1d and volume_ok[i]:
                signals[i] = -position_size[i]
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit: Price crosses below S3 OR trend reverses
                if close[i] < s3_aligned[i] or close[i] < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size[i]
            elif position == -1:
                # Exit: Price crosses above R3 OR trend reverses
                if close[i] > r3_aligned[i] or close[i] > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size[i]
    
    return signals
```

## Last Updated
2026-05-11 09:32

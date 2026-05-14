# Strategy: 4H_Donchian20_1DTrend_Filter_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.245 | +28.6% | -6.6% | 157 | PASS |
| ETHUSDT | 0.486 | +39.0% | -5.5% | 150 | PASS |
| SOLUSDT | 0.297 | +37.6% | -12.1% | 145 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.259 | -1.3% | -5.3% | 69 | FAIL |
| ETHUSDT | 1.235 | +19.5% | -4.0% | 57 | PASS |
| SOLUSDT | 0.441 | +10.5% | -6.2% | 48 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4H_Donchian20_1DTrend_Filter_VolumeSpike
# Hypothesis: 4-hour Donchian(20) breakout with daily trend filter (price > daily EMA34) and volume spike confirmation.
# Uses daily trend to avoid counter-trend trades in both bull and bear markets.
# Volume spike ensures momentum confirmation. Targets 20-40 trades/year to minimize fee drag.
# Uses discrete position sizing (0.25).

name = "4H_Donchian20_1DTrend_Filter_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough data for EMA34
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    lookback = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(lookback, n):
        donchian_high[i] = np.max(high[i-lookback:i])
        donchian_low[i] = np.min(low[i-lookback:i])
    
    # Volume filter: current volume > 2.0x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volatility filter: avoid low volatility periods (ATR < 0.3% of price)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    vol_filter = atr > 0.003 * close  # ATR > 0.3% of price
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Ensure we have Donchian and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(vol_filter[i]) or not vol_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation (2.0x average volume)
        volume_filter = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high + daily uptrend + volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > ema_34_1d_aligned[i] and   # Daily uptrend filter
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + daily downtrend + volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_34_1d_aligned[i] and   # Daily downtrend filter
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Price returns to the middle of Donchian channel (mean reversion)
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
            at_mid = abs(close[i] - donchian_mid) < (donchian_high[i] - donchian_low[i]) * 0.25  # Within 25% of range
            
            if at_mid:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals
```

## Last Updated
2026-05-07 03:38

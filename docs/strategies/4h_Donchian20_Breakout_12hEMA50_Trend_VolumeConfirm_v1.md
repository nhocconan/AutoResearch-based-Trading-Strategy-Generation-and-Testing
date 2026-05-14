# Strategy: 4h_Donchian20_Breakout_12hEMA50_Trend_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.042 | +19.8% | -23.9% | 134 | PASS |
| ETHUSDT | 0.086 | +21.4% | -19.1% | 124 | PASS |
| SOLUSDT | 0.969 | +225.0% | -30.3% | 130 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.232 | -10.2% | -12.8% | 54 | FAIL |
| ETHUSDT | 0.221 | +9.4% | -11.1% | 43 | PASS |
| SOLUSDT | 0.591 | +19.5% | -13.1% | 43 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend + volume confirmation (>1.5x 20-bar MA)
# Donchian breakout captures momentum, 12h EMA50 filters trend direction, volume confirms strength.
# Works in bull markets via breakouts and in bear markets via short breakdowns with trend filter.
# Target: 100-200 total trades over 4 years (25-50/year) with discrete sizing (0.30).

name = "4h_Donchian20_Breakout_12hEMA50_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h HTF data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(lookback, 20)  # Need 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_4h[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(volume_ma_20[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper band, above 12h EMA50, and volume confirmation
            if curr_close > highest_high[i-1] and curr_close > ema_50_4h[i] and vol_confirm:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Donchian lower band, below 12h EMA50, and volume confirmation
            elif curr_close < lowest_low[i-1] and curr_close < ema_50_4h[i] and vol_confirm:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on price breaking below Donchian lower band or below 12h EMA50
            if curr_close < lowest_low[i-1] or curr_close < ema_50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit on price breaking above Donchian upper band or above 12h EMA50
            if curr_close > highest_high[i-1] or curr_close > ema_50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
```

## Last Updated
2026-05-01 16:39

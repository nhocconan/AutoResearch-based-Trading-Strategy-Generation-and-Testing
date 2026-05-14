# Strategy: 4h_Donchian20_Breakout_1dTrend_VolumeSpike_ATRStop_v3

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.183 | +27.9% | -11.5% | 199 | PASS |
| ETHUSDT | -0.052 | +17.5% | -10.1% | 180 | FAIL |
| SOLUSDT | 0.569 | +67.7% | -14.3% | 161 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.094 | +6.8% | -5.0% | 70 | PASS |
| SOLUSDT | 0.979 | +20.1% | -6.4% | 50 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeSpike_ATRStop_v3
Hypothesis: Trade Donchian(20) breakouts with 1d EMA50 trend filter and volume confirmation (2.0x average) for fewer trades. Uses ATR trailing stop (2.5). Designed for very low trade frequency (~15-30/year) by requiring strong confluence: breakout + HTF trend + volume spike. Works in bull markets (breakouts with trend) and bear markets (short breakdowns against trend). Focus on BTC/ETH as primary targets.
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
    
    # Get 1d data for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter (slower = fewer whipsaws)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Donchian(20) on 4h: highest high/lowest low of past 20 bars (including current?)
    # We use past 20 completed bars, so lookback 20, exclude current bar
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align 1d indicators to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 2.0x average volume (tighter for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop (14-period on 4h)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Warmup: max of 1d EMA (50), Donchian lookback (20), volume MA (20), 4h ATR (14)
    start_idx = max(50, 20, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr_14[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        highest_20_val = highest_20[i]
        lowest_20_val = lowest_20[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        atr_14_val = atr_14[i]
        
        if position == 0:
            # Long: break above highest_20, uptrend (close > EMA50), volume spike
            long_signal = (high_val > highest_20_val) and (close_val > ema_50_1d_val) and (volume_val > 2.0 * vol_ma_val)
            # Short: break below lowest_20, downtrend (close < EMA50), volume spike
            short_signal = (low_val < lowest_20_val) and (close_val < ema_50_1d_val) and (volume_val > 2.0 * vol_ma_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                long_stop = entry_price - 2.5 * atr_14_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                short_stop = entry_price + 2.5 * atr_14_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Update trailing stop: move stop up as price makes new highs
            long_stop = max(long_stop, high_val - 2.5 * atr_14_val)
            # Exit: trailing stop hit or trend reversal (close < EMA50)
            if (low_val < long_stop) or (close_val < ema_50_1d_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, low_val + 2.5 * atr_14_val)
            # Exit: trailing stop hit or trend reversal (close > EMA50)
            if (high_val > short_stop) or (close_val > ema_50_1d_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeSpike_ATRStop_v3"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-26 01:44

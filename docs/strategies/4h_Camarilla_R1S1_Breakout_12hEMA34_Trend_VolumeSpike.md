# Strategy: 4h_Camarilla_R1S1_Breakout_12hEMA34_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.085 | +15.2% | -15.0% | 242 | FAIL |
| ETHUSDT | 0.077 | +22.8% | -15.6% | 239 | PASS |
| SOLUSDT | 0.782 | +115.7% | -21.0% | 209 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.483 | +13.7% | -12.6% | 82 | PASS |
| SOLUSDT | 0.024 | +5.5% | -11.9% | 74 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_12hEMA34_Trend_VolumeSpike
Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA34 trend filter and volume spike confirmation.
Goes long when price breaks above R1 with 12h uptrend (price > EMA34) and volume > 1.8x 20-period average,
short when price breaks below S1 with 12h downtrend (price < EMA34) and volume > 1.8x 20-period average.
Exit on opposite Camarilla level touch or trend reversal. Uses discrete sizing (0.25) to minimize fees.
Target: 20-40 trades/year. Works in bull via breakouts with trend, in bear via mean reversion at extremes.
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
    
    # Get 1d data for Camarilla calculations (using daily OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for today using yesterday's OHLC
    prev_close = np.concatenate([[close_1d[0]], close_1d[:-1]])  # yesterday's close
    prev_high = np.concatenate([[high_1d[0]], high_1d[:-1]])   # yesterday's high
    prev_low = np.concatenate([[low_1d[0]], low_1d[:-1]])     # yesterday's low
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + 0.275 * camarilla_range
    s1 = prev_close - 0.275 * camarilla_range
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 12h EMA34 for trend
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above R1, 12h uptrend (price > EMA34), volume spike
            long_signal = (close[i] > r1_aligned[i]) and (close[i] > ema_34_12h_aligned[i]) and vol_spike[i]
            # Short: price breaks below S1, 12h downtrend (price < EMA34), volume spike
            short_signal = (close[i] < s1_aligned[i]) and (close[i] < ema_34_12h_aligned[i]) and vol_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price closes below S1 (mean reversion) or 12h trend turns down
            exit_signal = (close[i] < s1_aligned[i]) or (close[i] < ema_34_12h_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price closes above R1 (mean reversion) or 12h trend turns up
            exit_signal = (close[i] > r1_aligned[i]) or (close[i] > ema_34_12h_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 20:10

# Strategy: 4h_WilliamsFractal_Breakout_1dEMA34_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.402 | +38.9% | -10.5% | 48 | PASS |
| ETHUSDT | 0.537 | +52.0% | -9.4% | 47 | PASS |
| SOLUSDT | 1.131 | +181.8% | -20.1% | 51 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.402 | -3.3% | -6.0% | 24 | FAIL |
| ETHUSDT | 0.640 | +13.7% | -5.8% | 14 | PASS |
| SOLUSDT | -0.018 | +5.3% | -6.5% | 15 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h Williams Fractal Breakout + Daily EMA34 Trend + Volume Spike
Hypothesis: Williams fractals on 1d identify key swing levels. Breakouts above 
bearish fractal (resistance) or below bullish fractal (support) with daily EMA34 
trend alignment and volume spike capture institutional participation. Works in 
bull markets (trend continuation) and bear markets (failed breaks, reversals to 
fractal levels). 4h timeframe targets 20-50 trades/year to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_ = prices['open'].values
    
    # Daily data for Williams fractals and EMA34 (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Williams fractals: needs 2 extra bars for confirmation
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Align with additional_delay_bars=2 for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Daily EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for daily data (fractals + EMA) and volume MA
    start_idx = max(34, 20) + 10  # extra for fractal confirmation delay
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        vol_spike = volume_spike[i]
        
        # Breakout conditions
        breakout_long = curr_close > bearish_fractal_aligned[i]
        breakout_short = curr_close < bullish_fractal_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: fractal breakout + volume spike + daily EMA34 trend alignment
            long_entry = breakout_long and vol_spike and (curr_close > ema_34_1d_aligned[i])
            short_entry = breakout_short and vol_spike and (curr_close < ema_34_1d_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit on bullish fractal retrace or trend change
            if curr_close < bullish_fractal_aligned[i] or curr_close < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on bearish fractal retrace or trend change
            if curr_close > bearish_fractal_aligned[i] or curr_close > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsFractal_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 08:23

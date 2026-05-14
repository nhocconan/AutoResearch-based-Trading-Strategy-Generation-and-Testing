# Strategy: 12h_Williams_Fractal_Breakout_1dEMA34_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.185 | +27.7% | -8.9% | 24 | PASS |
| ETHUSDT | 0.203 | +30.0% | -10.8% | 22 | PASS |
| SOLUSDT | 1.178 | +165.1% | -18.5% | 20 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.948 | -4.3% | -5.2% | 10 | FAIL |
| ETHUSDT | 0.320 | +9.4% | -6.2% | 5 | PASS |
| SOLUSDT | -1.227 | -7.6% | -13.7% | 6 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
12h Williams Fractal Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Williams fractals identify key swing points on 1d timeframe. Breakouts above/below
these levels capture momentum in the direction of the 1d EMA34 trend. Volume spike confirms
participation. Designed for 12h timeframe with tight entry conditions to achieve 12-37 trades/year.
Works in bull (breakouts above fractal highs in uptrend) and bear (breakouts below fractal lows in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams fractals and EMA (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams fractals on 1d high/low (requires 5-bar window: 2 left, 2 right)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Williams fractals need 2 extra 1d bars after center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Calculate EMA34 on 1d close for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for fractals (5-bar lookback), EMA, volume MA
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        bullish_fractal_level = bullish_fractal_aligned[i]
        bearish_fractal_level = bearish_fractal_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above bullish fractal high AND volume spike AND price > EMA (uptrend)
            long_entry = (curr_high > bullish_fractal_level) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below bearish fractal low AND volume spike AND price < EMA (downtrend)
            short_entry = (curr_low < bearish_fractal_level) and vol_spike and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below bearish fractal low OR price crosses below EMA (trend change)
            if (curr_low < bearish_fractal_level) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above bullish fractal high OR price crosses above EMA (trend change)
            if (curr_high > bullish_fractal_level) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Fractal_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-25 05:37

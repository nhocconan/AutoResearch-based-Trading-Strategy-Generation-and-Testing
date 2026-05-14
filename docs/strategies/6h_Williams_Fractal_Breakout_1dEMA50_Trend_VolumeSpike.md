# Strategy: 6h_Williams_Fractal_Breakout_1dEMA50_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.404 | +39.0% | -9.7% | 37 | PASS |
| ETHUSDT | 0.100 | +24.6% | -13.3% | 37 | PASS |
| SOLUSDT | 1.178 | +186.7% | -20.4% | 32 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.806 | +0.4% | -6.4% | 15 | FAIL |
| ETHUSDT | 0.673 | +14.4% | -7.5% | 10 | PASS |
| SOLUSDT | -0.485 | -0.7% | -8.0% | 11 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h Williams Fractal Breakout + 1d EMA50 Trend + Volume Spike
Hypothesis: Williams fractals identify key swing points. Breakouts above recent bullish fractals or below bearish fractals,
when aligned with 1d EMA50 trend and confirmed by volume spikes, capture momentum moves with low false breakout rate.
Fractals require 2-bar confirmation, reducing whipsaws. Designed for 6h to target 12-37 trades/year (50-150 over 4 years)
by requiring confluence of fractal breakout, trend alignment, and volume confirmation, minimizing fee drag in bear markets.
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
    
    # Load 1d data ONCE before loop for indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Compute Williams fractals on 1d (requires 5 bars: n-2, n-1, n, n+1, n+2)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Align fractals to 6h with 2-bar extra delay for confirmation (fractal needs 2 future bars to confirm)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # 1d EMA50 for trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average (stricter for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 50, 2)  # volume MA, EMA50, fractal alignment
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1d EMA50
        bullish_bias = curr_close > ema_1d_aligned[i]
        bearish_bias = curr_close < ema_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals - require ALL conditions: fractal breakout + trend + volume
            # Long: price breaks above recent bullish fractal AND bullish bias AND volume spike
            long_entry = (curr_high > bullish_fractal_aligned[i]) and bullish_bias and vol_spike
            # Short: price breaks below recent bearish fractal AND bearish bias AND volume spike
            short_entry = (curr_low < bearish_fractal_aligned[i]) and bearish_bias and vol_spike
            
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
            # Exit: price falls below recent bearish fractal (mean reversion) OR loss of bullish bias
            if (curr_low < bearish_fractal_aligned[i]) or (curr_close < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above recent bullish fractal (mean reversion) OR loss of bearish bias
            if (curr_high > bullish_fractal_aligned[i]) or (curr_close > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Williams_Fractal_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-25 07:13

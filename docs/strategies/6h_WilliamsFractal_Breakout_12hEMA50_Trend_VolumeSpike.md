# Strategy: 6h_WilliamsFractal_Breakout_12hEMA50_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.389 | +36.1% | -8.4% | 96 | PASS |
| ETHUSDT | 0.636 | +52.9% | -6.6% | 77 | PASS |
| SOLUSDT | 1.119 | +147.2% | -28.2% | 77 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.211 | -1.9% | -8.3% | 33 | FAIL |
| ETHUSDT | 0.183 | +7.9% | -6.9% | 27 | PASS |
| SOLUSDT | -0.894 | -2.8% | -10.7% | 25 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h Williams Fractal Breakout with 12h EMA Trend Filter and Volume Confirmation
Hypothesis: Williams fractals identify key swing points where institutional order flow accumulates.
Breakouts above recent bearish fractals or below bullish fractals with volume confirmation
and aligned with 12h EMA50 trend capture momentum moves in both bull and bear markets.
ATR-based trailing stop manages risk. Target: 12-30 trades/year on 6h (50-120 total over 4 years)
to minimize fee drag while maintaining edge in ranging and trending conditions.
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
    
    # Williams Fractals on 6h data (primary timeframe)
    bearish_fractal, bullish_fractal = compute_williams_fractals(high, low)
    # Fractals need 2-bar confirmation delay (forming fractal + 2 more bars to confirm)
    bearish_fractal_aligned = align_htf_to_ltf(prices, prices, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, prices, bullish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: current volume > 1.8 * 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    # ATR for volatility filter and trailing stop
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 12h EMA50 trend filter (MTF)
    df_12h = get_htf_data(prices, '12h')
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for all indicators
    start_idx = max(30, 50) + 5  # extra buffer for fractal confirmation
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr_14[i]) or
            np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Breakout conditions: price breaks recent confirmed fractal levels
        breakout_long = curr_close > bearish_fractal_aligned[i]
        breakout_short = curr_close < bullish_fractal_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: fractal breakout + volume spike + 12h EMA trend alignment
            long_entry = breakout_long and vol_spike and (curr_close > ema_50_12h_aligned[i])
            short_entry = breakout_short and vol_spike and (curr_close < ema_50_12h_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
            elif short_entry:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_low
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management: ATR trailing stop
            highest_since_entry = max(highest_since_entry, curr_high)
            exit_level = highest_since_entry - (2.5 * atr_14[i])
            
            if curr_close < exit_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management: ATR trailing stop
            lowest_since_entry = min(lowest_since_entry, curr_low)
            exit_level = lowest_since_entry + (2.5 * atr_14[i])
            
            if curr_close > exit_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsFractal_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-25 08:12

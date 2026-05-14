# Strategy: 6h_Williams_Fractal_Breakout_12hEMA34_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.236 | +29.4% | -8.6% | 94 | PASS |
| ETHUSDT | 0.349 | +35.7% | -7.5% | 75 | PASS |
| SOLUSDT | 1.427 | +178.5% | -12.1% | 68 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.439 | -2.7% | -7.5% | 38 | FAIL |
| ETHUSDT | 0.434 | +10.7% | -6.9% | 28 | PASS |
| SOLUSDT | -1.145 | -5.3% | -12.5% | 30 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h Williams Fractal Breakout + 12h EMA34 Trend + Volume Spike
Hypothesis: Williams fractals identify swing highs/lows with confirmation delay; breakouts from these levels with 12h EMA trend filter and volume spike capture strong momentum moves. Works in both bull (buy breakouts above bearish fractals in uptrend) and bear (sell breakdowns below bullish fractals in downtrend) via symmetric logic. Target 12-37 trades/year on 6h to avoid fee drag.
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
    
    # Get 12h data for EMA34 trend filter (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    close_12h = pd.Series(df_12h['close'])
    ema_34_12h = close_12h.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Get 1d data for Williams fractals (need 2 extra bars for confirmation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Williams fractals on 1d
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values
    )
    # Align with 2 extra delay bars for fractal confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate ATR(14) for stop management
    atr = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for EMA34, ATR, volume MA, fractals
    start_idx = max(34, 14, 20, 50)  # 50 for fractal stability
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_34_val = ema_34_12h_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        bearish_fractal_val = bearish_fractal_aligned[i]
        bullish_fractal_val = bullish_fractal_aligned[i]
        
        # Trend filter: price relative to 12h EMA34
        uptrend = curr_close > ema_34_val
        downtrend = curr_close < ema_34_val
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for breakout signals at fractal levels
            # Long: price breaks above bearish fractal with volume confirmation in uptrend
            long_breakout = (curr_close > bearish_fractal_val) and volume_confirm and uptrend
            # Short: price breaks below bullish fractal with volume confirmation in downtrend
            short_breakout = (curr_close < bullish_fractal_val) and volume_confirm and downtrend
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit conditions: price closes below bullish fractal OR 2.5*ATR trailing stop OR EMA34 trend turns down
            if curr_close < bullish_fractal_val or curr_close < (highest_since_entry - 2.5 * atr_val) or curr_close < ema_34_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit conditions: price closes above bearish fractal OR 2.5*ATR trailing stop OR EMA34 trend turns up
            if curr_close > bearish_fractal_val or curr_close > (lowest_since_entry + 2.5 * atr_val) or curr_close > ema_34_val:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Williams_Fractal_Breakout_12hEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-25 04:12

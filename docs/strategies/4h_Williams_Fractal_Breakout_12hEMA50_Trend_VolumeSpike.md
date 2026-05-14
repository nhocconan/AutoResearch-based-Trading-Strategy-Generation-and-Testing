# Strategy: 4h_Williams_Fractal_Breakout_12hEMA50_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.147 | +26.2% | -9.1% | 134 | PASS |
| ETHUSDT | 0.111 | +25.0% | -10.6% | 119 | PASS |
| SOLUSDT | 0.865 | +99.0% | -20.0% | 110 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.061 | -6.3% | -6.9% | 53 | FAIL |
| ETHUSDT | 1.155 | +18.7% | -4.1% | 35 | PASS |
| SOLUSDT | -0.333 | +2.3% | -8.2% | 36 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h Williams Fractal Breakout + 12h EMA50 Trend + Volume Spike
Hypothesis: Williams fractals on 12h identify major swing points with 2-bar confirmation delay. 
Breakouts above/below these levels with 12h EMA50 trend filter and volume spike capture strong momentum moves. 
Works in bull (buy breakouts above bearish fractals in uptrend) and bear (sell breakdowns below bullish fractals in downtrend) via symmetric logic. 
Target 20-50 trades/year on 4h to avoid fee drag.
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
    
    # Get 12h data for EMA50 trend filter (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = pd.Series(df_12h['close'])
    ema_50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 12h data for Williams fractals (need 2 extra bars for confirmation)
    if len(df_12h) < 5:
        return np.zeros(n)
    
    # Calculate Williams fractals on 12h
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_12h['high'].values,
        df_12h['low'].values
    )
    # Align with 2 extra delay bars for fractal confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bullish_fractal, additional_delay_bars=2)
    
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
    
    # Start index: need enough for EMA50, ATR, volume MA, fractals
    start_idx = max(50, 14, 20, 50)  # 50 for EMA50 and fractal stability
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i]) or 
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
        ema_50_val = ema_50_12h_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        bearish_fractal_val = bearish_fractal_aligned[i]
        bullish_fractal_val = bullish_fractal_aligned[i]
        
        # Trend filter: price relative to 12h EMA50
        uptrend = curr_close > ema_50_val
        downtrend = curr_close < ema_50_val
        
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
            # Exit conditions: price closes below bullish fractal OR 2.5*ATR trailing stop OR EMA50 trend turns down
            if curr_close < bullish_fractal_val or curr_close < (highest_since_entry - 2.5 * atr_val) or curr_close < ema_50_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit conditions: price closes above bearish fractal OR 2.5*ATR trailing stop OR EMA50 trend turns up
            if curr_close > bearish_fractal_val or curr_close > (lowest_since_entry + 2.5 * atr_val) or curr_close > ema_50_val:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Williams_Fractal_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 04:19

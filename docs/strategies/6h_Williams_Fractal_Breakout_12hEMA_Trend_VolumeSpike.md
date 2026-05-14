# Strategy: 6h_Williams_Fractal_Breakout_12hEMA_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.020 | +21.3% | -13.1% | 66 | PASS |
| ETHUSDT | 0.359 | +37.8% | -9.1% | 50 | PASS |
| SOLUSDT | 1.386 | +212.8% | -12.9% | 54 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.226 | -2.4% | -6.5% | 27 | FAIL |
| ETHUSDT | 0.206 | +8.2% | -6.9% | 22 | PASS |
| SOLUSDT | -1.079 | -6.9% | -13.6% | 26 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h Williams Fractal Breakout + 12h EMA Trend + Volume Spike
Hypothesis: Williams fractals identify key swing points. Breakouts above recent bearish fractals or below bullish fractals with 12h EMA trend alignment and volume spikes capture strong moves. Works in bull (long on upside breaks) and bear (short on downside breaks). Targets 50-150 trades over 4 years on 6h timeframe.
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
    
    # Get 12h data for EMA trend (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 12h
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Get 1d data for Williams fractals (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Williams fractals on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    # Bearish fractal needs 2 extra 1d bars for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 20-period volume MA for volume spike
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for fractals (5) and volume MA
    start_idx = max(5, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_34_12h_aligned[i]
        bearish_fractal_val = bearish_fractal_aligned[i]
        bullish_fractal_val = bullish_fractal_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above bearish fractal, above 12h EMA, volume confirmation
            long_entry = (curr_close > bearish_fractal_val and 
                         curr_close > ema_trend and 
                         volume_confirm)
            # Short: price breaks below bullish fractal, below 12h EMA, volume confirmation
            short_entry = (curr_close < bullish_fractal_val and 
                          curr_close < ema_trend and 
                          volume_confirm)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below bullish fractal OR below 12h EMA
            if curr_close < bullish_fractal_val or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above bearish fractal OR above 12h EMA
            if curr_close > bearish_fractal_val or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Williams_Fractal_Breakout_12hEMA_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-25 04:41

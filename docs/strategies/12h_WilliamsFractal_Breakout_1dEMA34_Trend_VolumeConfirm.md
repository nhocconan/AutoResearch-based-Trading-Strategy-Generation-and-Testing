# Strategy: 12h_WilliamsFractal_Breakout_1dEMA34_Trend_VolumeConfirm

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.037 | +19.1% | -13.5% | 28 | FAIL |
| ETHUSDT | 0.059 | +22.4% | -10.8% | 27 | PASS |
| SOLUSDT | 1.314 | +199.0% | -17.8% | 24 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.501 | +11.8% | -6.2% | 6 | PASS |
| SOLUSDT | -0.489 | -0.2% | -10.2% | 7 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Fractal Breakout with 1d EMA34 Trend and Volume Confirmation
- Williams Fractals identify key swing highs/lows as potential breakout levels
- Breakout above recent bearish fractal or below bullish fractal with volume confirmation captures momentum
- 1d EMA(34) ensures alignment with daily trend
- Designed for 12h timeframe targeting 12-37 trades/year (50-150 over 4 years) to minimize fee drag
- Works in both bull and bear markets by trading breakouts in direction of 1d trend
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
    
    # Get 1d data for Williams Fractals and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Fractals on 1d
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    # Williams fractals need 2 extra 1d bars for confirmation (center bar + 2 right bars)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: > 1.8x 30-period average on 12h timeframe
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 30)  # EMA1d, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Fractal breakout signals with trend filter and volume confirmation
        # Long: price breaks above recent bearish fractal (resistance) + uptrend + volume spike
        # Short: price breaks below recent bullish fractal (support) + downtrend + volume spike
        long_signal = (close[i] > bearish_fractal_aligned[i] and 
                      close[i] > ema_34_1d_aligned[i] and
                      volume[i] > 1.8 * vol_ma[i])
        
        short_signal = (close[i] < bullish_fractal_aligned[i] and 
                       close[i] < ema_34_1d_aligned[i] and
                       volume[i] > 1.8 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: trend reversal or opposite fractal level break
            exit_signal = False
            
            if position == 1:
                # Exit long: trend reversal or price breaks below bullish fractal (support)
                if (close[i] < ema_34_1d_aligned[i] or 
                    close[i] < bullish_fractal_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: trend reversal or price breaks above bearish fractal (resistance)
                if (close[i] > ema_34_1d_aligned[i] or 
                    close[i] > bearish_fractal_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsFractal_Breakout_1dEMA34_Trend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-23 17:56

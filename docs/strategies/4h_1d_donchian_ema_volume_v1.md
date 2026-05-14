# Strategy: 4h_1d_donchian_ema_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.109 | +25.0% | -12.2% | 116 | KEEP |
| ETHUSDT | -0.481 | -11.4% | -24.4% | 125 | DISCARD |
| SOLUSDT | 0.526 | +79.8% | -29.7% | 117 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.082 | -5.3% | -7.5% | 44 | DISCARD |
| SOLUSDT | 0.327 | +11.5% | -14.5% | 37 | KEEP |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter (EMA50) and volume confirmation
# Uses Donchian channels from 4h data: breakout above upper band = long, below lower band = short
# 1d EMA50 filter ensures trades align with higher timeframe trend
# Volume confirmation reduces false breakouts
# Designed for 4h timeframe to target 20-50 trades/year (75-200 over 4 years)
# Works in bull/bear: EMA50 adapts to trend, Donchian provides structure

name = "4h_1d_donchian_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 with proper min_periods
    close_1d = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period Donchian channels on 4h data
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            upper_channel[i] = np.nan
            lower_channel[i] = np.nan
        else:
            upper_channel[i] = np.max(high[i-20:i])
            lower_channel[i] = np.min(low[i-20:i])
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or
            np.isnan(ema_50_4h[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price closes below lower Donchian band OR trend turns bearish
            if close[i] < lower_channel[i] or close[i] < ema_50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above upper Donchian band OR trend turns bullish
            if close[i] > upper_channel[i] or close[i] > ema_50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation
            if volume_confirm:
                # Long breakout: price closes above upper Donchian band AND price > 1d EMA50 (bullish trend)
                if close[i] > upper_channel[i] and close[i] > ema_50_4h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price closes below lower Donchian band AND price < 1d EMA50 (bearish trend)
                elif close[i] < lower_channel[i] and close[i] < ema_50_4h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-11 00:12

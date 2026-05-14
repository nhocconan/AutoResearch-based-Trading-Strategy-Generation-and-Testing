# Strategy: 4h_12h_donchian_hma_volume_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.250 | +30.8% | -8.9% | 142 | KEEP |
| ETHUSDT | 0.231 | +32.0% | -10.6% | 133 | KEEP |
| SOLUSDT | 0.650 | +82.3% | -18.1% | 142 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.311 | -4.2% | -6.1% | 48 | DISCARD |
| ETHUSDT | -0.138 | +3.7% | -10.3% | 49 | DISCARD |
| SOLUSDT | 0.418 | +11.4% | -6.4% | 42 | KEEP |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter (HMA21) and volume confirmation
# Uses Donchian channels from 4h data: breakout above upper band = long, below lower band = short
# 12h HMA21 filter ensures trades align with higher timeframe trend
# Volume confirmation reduces false breakouts
# Designed for 4h timeframe to target 20-50 trades/year (75-200 over 4 years)
# Works in bull/bear: HMA21 adapts to trend, Donchian provides robust structure

name = "4h_12h_donchian_hma_volume_v2"
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
    
    # Load 12h data ONCE before loop for Donchian channels and HMA21
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) from 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Upper band: highest high of last 20 periods
    upper_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 periods
    lower_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align 12h Donchian bands to 4h timeframe
    upper_20_4h = align_htf_to_ltf(prices, df_12h, upper_20)
    lower_20_4h = align_htf_to_ltf(prices, df_12h, lower_20)
    
    # Calculate 12h HMA21 trend filter
    half_n = int(21/2 + 0.5)
    wma_half = pd.Series(close_12h).rolling(window=half_n, min_periods=half_n).mean()
    wma_full = pd.Series(close_12h).rolling(window=21, min_periods=21).mean()
    hma_21_12h = (2 * wma_half - wma_full).values
    
    # Align 12h HMA21 to 4h timeframe
    hma_21_4h = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    
    # Calculate 20-period average volume for volume confirmation (4h volume)
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
        if (np.isnan(upper_20_4h[i]) or np.isnan(lower_20_4h[i]) or
            np.isnan(hma_21_4h[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band OR trend turns bearish
            if close[i] < lower_20_4h[i] or close[i] < hma_21_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band OR trend turns bullish
            if close[i] > upper_20_4h[i] or close[i] > hma_21_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation
            if volume_confirm:
                # Long breakout: price closes above Donchian upper band AND price > 12h HMA21 (bullish trend)
                if close[i] > upper_20_4h[i] and close[i] > hma_21_4h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price closes below Donchian lower band AND price < 12h HMA21 (bearish trend)
                elif close[i] < lower_20_4h[i] and close[i] < hma_21_4h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-09 11:42

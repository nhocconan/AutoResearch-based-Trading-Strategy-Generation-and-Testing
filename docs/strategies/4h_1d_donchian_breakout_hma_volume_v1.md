# Strategy: 4h_1d_donchian_breakout_hma_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.594 | -4.7% | -22.1% | 198 | FAIL |
| ETHUSDT | 0.405 | +44.7% | -12.0% | 183 | PASS |
| SOLUSDT | 0.533 | +71.5% | -25.5% | 173 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.020 | +5.6% | -11.3% | 71 | PASS |
| SOLUSDT | 0.477 | +13.5% | -10.0% | 57 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + 1d HMA(21) trend + volume confirmation
    # Long when: price breaks above Donchian upper (20) AND price > 1d HMA21 AND volume > 1.8x 20-bar avg volume
    # Short when: price breaks below Donchian lower (20) AND price < 1d HMA21 AND volume > 1.8x 20-bar avg volume
    # Exit when: price crosses Donchian midpoint OR adverse 1d HMA21 crossover
    # Uses discrete sizing (0.25) targeting 75-200 trades over 4 years.
    # Donchian breakout captures momentum; 1d HMA21 filters counter-trend moves; volume reduces false breakouts.
    # Works in bull (trend-following breaks) and bear (mean-reversion exits at midpoint).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HMA(21) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 1d HMA(21): WMA(2*WMA(n/2) - WMA(n)), sqrt(n) period
    half = 21 // 2
    sqrt_n = int(np.sqrt(21))
    wma_half = pd.Series(close_1d).ewm(span=half, adjust=False).mean().values
    wma_full = pd.Series(close_1d).ewm(span=21, adjust=False).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma_1d = pd.Series(raw_hma).ewm(span=sqrt_n, adjust=False).mean().values
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Calculate volume confirmation: volume > 1.8x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.8 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(hma_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_upper[i-1]  # break above previous upper
        breakout_down = close[i] < donchian_lower[i-1]  # break below previous lower
        
        # 1d HMA21 trend filter
        uptrend = close[i] > hma_1d_aligned[i]
        downtrend = close[i] < hma_1d_aligned[i]
        
        # Entry conditions with volume confirmation
        long_entry = breakout_up and uptrend and volume_confirmed[i] and position != 1
        short_entry = breakout_down and downtrend and volume_confirmed[i] and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and (close[i] < donchian_mid[i] or not uptrend))
        exit_short = (position == -1 and (close[i] > donchian_mid[i] or not downtrend))
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_breakout_hma_volume_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-13 07:16

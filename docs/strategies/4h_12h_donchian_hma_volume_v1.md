# Strategy: 4h_12h_donchian_hma_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.148 | +12.7% | -17.8% | 237 | FAIL |
| ETHUSDT | 0.059 | +21.9% | -11.1% | 225 | PASS |
| SOLUSDT | 0.442 | +60.9% | -27.2% | 225 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.134 | +7.4% | -12.0% | 79 | PASS |
| SOLUSDT | 0.474 | +13.6% | -9.3% | 67 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 12h HMA(21) trend filter and volume confirmation
    # Long when: price breaks above Donchian(20) high AND price > 12h HMA(21) (uptrend) AND volume > 1.5x 20-bar avg volume
    # Short when: price breaks below Donchian(20) low AND price < 12h HMA(21) (downtrend) AND volume > 1.5x 20-bar avg volume
    # Exit when: price crosses Donchian(20) midpoint OR adverse 12h HMA(21) crossover
    # Uses discrete sizing (0.25) targeting 75-200 trades over 4 years.
    # Works in bull/bear via 12h HMA(21) trend filter preventing counter-trend trades.
    # Volume confirmation reduces false breakouts in choppy markets.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for HMA(21) trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate 12h HMA(21)
    half_length = 21 // 2
    sqrt_length = int(np.sqrt(21))
    wma1 = pd.Series(close_12h).ewm(span=half_length, adjust=False, min_periods=half_length).mean().values
    wma2 = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    raw_hma = 2 * wma1 - wma2
    hma_12h = pd.Series(raw_hma).ewm(span=sqrt_length, adjust=False, min_periods=sqrt_length).mean().values
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Calculate volume confirmation: volume > 1.5x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(hma_12h_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i-1]  # Break above previous period high
        breakout_down = close[i] < lowest_low[i-1]  # Break below previous period low
        
        # 12h HMA(21) trend filter
        uptrend = close[i] > hma_12h_aligned[i]
        downtrend = close[i] < hma_12h_aligned[i]
        
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

name = "4h_12h_donchian_hma_volume_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-13 07:12

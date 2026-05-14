# Strategy: 6h_12h_donchian_breakout_ema_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.212 | +10.5% | -13.5% | 145 | FAIL |
| ETHUSDT | 0.441 | +47.1% | -9.7% | 135 | PASS |
| SOLUSDT | 0.923 | +137.1% | -15.4% | 121 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.342 | +10.9% | -9.8% | 41 | PASS |
| SOLUSDT | 0.255 | +9.5% | -8.7% | 43 | PASS |

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
    
    # Hypothesis: 6h Donchian(20) breakout + 12h trend filter (EMA50) + volume confirmation
    # Long when: price breaks above Donchian upper (20) AND price > 12h EMA50 AND volume > 1.5x 20-bar avg volume
    # Short when: price breaks below Donchian lower (20) AND price < 12h EMA50 AND volume > 1.5x 20-bar avg volume
    # Exit when: price crosses Donchian midpoint (10-bar avg of upper/lower) OR adverse 12h EMA50 crossover
    # Uses discrete sizing (0.25) targeting 50-150 trades over 4 years.
    # Donchian breakout captures momentum; 12h EMA50 filters counter-trend moves; volume reduces false breakouts.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate 12h EMA(50)
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate Donchian channels (20-period) on 6h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Calculate volume confirmation: volume > 1.5x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(ema_12h_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_upper[i-1]  # break above previous upper
        breakout_down = close[i] < donchian_lower[i-1]  # break below previous lower
        
        # 12h EMA50 trend filter
        uptrend = close[i] > ema_12h_aligned[i]
        downtrend = close[i] < ema_12h_aligned[i]
        
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

name = "6h_12h_donchian_breakout_ema_volume_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-13 07:16

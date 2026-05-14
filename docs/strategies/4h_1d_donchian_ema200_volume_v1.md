# Strategy: 4h_1d_donchian_ema200_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.292 | +30.8% | -11.4% | 97 | PASS |
| ETHUSDT | 0.074 | +23.5% | -10.3% | 84 | PASS |
| SOLUSDT | 0.940 | +96.1% | -9.4% | 77 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.258 | -3.1% | -6.8% | 40 | FAIL |
| ETHUSDT | 0.248 | +8.9% | -8.0% | 37 | PASS |
| SOLUSDT | -0.121 | +3.9% | -9.7% | 32 | FAIL |

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
    
    # Hypothesis: 4h Donchian(20) breakout + 1d EMA200 trend filter + volume confirmation
    # Long when: price breaks above 4h Donchian upper (20) AND price > 1d EMA200 AND volume > 2x 20-bar avg
    # Short when: price breaks below 4h Donchian lower (20) AND price < 1d EMA200 AND volume > 2x 20-bar avg
    # Exit when: price crosses 4h Donchian midpoint
    # Uses discrete sizing (0.25) targeting 100-200 total trades over 4 years.
    # Donchian provides structure; 1d EMA200 filters counter-trend breaks; volume confirms validity.
    # Works in bull (breakouts with trend) and bear (strong trend-aligned breaks only).
    # 1d EMA200 is a strong trend filter that reduces whipsaw in sideways markets.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA200
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate volume confirmation: volume > 2x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(donchian_window, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions (using current bar's close vs previous bar's levels)
        breakout_up = close[i] > donchian_high[i-1]  # break above previous Donchian high
        breakout_down = close[i] < donchian_low[i-1]  # break below previous Donchian low
        
        # Entry conditions with trend filter and volume confirmation
        long_entry = breakout_up and (close[i] > ema_200_1d_aligned[i]) and volume_confirmed[i] and position != 1
        short_entry = breakout_down and (close[i] < ema_200_1d_aligned[i]) and volume_confirmed[i] and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and close[i] < donchian_mid[i])
        exit_short = (position == -1 and close[i] > donchian_mid[i])
        
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

name = "4h_1d_donchian_ema200_volume_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-13 07:27

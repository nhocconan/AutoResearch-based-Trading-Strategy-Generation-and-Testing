# Strategy: 4h_12h_donchian_ema50_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.128 | +13.7% | -16.6% | 195 | FAIL |
| ETHUSDT | 0.363 | +42.0% | -12.9% | 178 | PASS |
| SOLUSDT | 0.606 | +83.9% | -27.6% | 185 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.593 | +15.0% | -6.8% | 57 | PASS |
| SOLUSDT | 0.262 | +9.7% | -9.6% | 56 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume confirmation
    # Long when: price breaks above 4h Donchian upper (20) AND price > 12h EMA50 AND volume > 1.5x 20-bar avg
    # Short when: price breaks below 4h Donchian lower (20) AND price < 12h EMA50 AND volume > 1.5x 20-bar avg
    # Exit when: price crosses 4h Donchian midpoint
    # Uses discrete sizing (0.25) targeting 75-200 total trades over 4 years (19-50/year).
    # 4h timeframe balances trade frequency and signal quality.
    # 12h EMA50 provides strong trend filter, reducing whipsaw in choppy/ranging markets.
    # Volume confirmation ensures breakouts have institutional conviction.
    # Works in bull (breakouts with trend) and bear (only trend-aligned breaks taken).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_window = 20
    donchian_high_4h = pd.Series(high_4h).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low_4h = pd.Series(low_4h).rolling(window=donchian_window, min_periods=donchian_window).min().values
    donchian_mid_4h = (donchian_high_4h + donchian_low_4h) / 2.0
    
    # Align 4h Donchian levels to 15m timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid_4h)
    
    # Get 12h data for EMA50 trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate volume confirmation: volume > 1.5x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(donchian_window, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(donchian_mid_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions (using current bar's close vs previous bar's levels)
        breakout_up = close[i] > donchian_high_aligned[i-1]  # break above previous Donchian high
        breakout_down = close[i] < donchian_low_aligned[i-1]  # break below previous Donchian low
        
        # Entry conditions with trend filter and volume confirmation
        long_entry = breakout_up and (close[i] > ema_50_12h_aligned[i]) and volume_confirmed[i] and position != 1
        short_entry = breakout_down and (close[i] < ema_50_12h_aligned[i]) and volume_confirmed[i] and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and close[i] < donchian_mid_aligned[i])
        exit_short = (position == -1 and close[i] > donchian_mid_aligned[i])
        
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

name = "4h_12h_donchian_ema50_volume_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-13 07:32

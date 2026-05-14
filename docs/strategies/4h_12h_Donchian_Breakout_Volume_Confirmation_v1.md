# Strategy: 4h_12h_Donchian_Breakout_Volume_Confirmation_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.125 | +14.0% | -13.4% | 196 | FAIL |
| ETHUSDT | 0.197 | +30.5% | -9.5% | 180 | PASS |
| SOLUSDT | 0.671 | +93.3% | -26.3% | 185 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.599 | +15.0% | -6.8% | 59 | PASS |
| SOLUSDT | 0.294 | +10.2% | -9.4% | 55 | PASS |

## Code
```python
# 4h_12h_Donchian_Breakout_Volume_Confirmation_v1
# Hypothesis: On 4h timeframe, buy when price breaks above Donchian(20) high with volume confirmation and 12h trend filter (price > 12h EMA50), sell when price breaks below Donchian(20) low with volume confirmation and 12h trend filter (price < 12h EMA50).
# Uses 12h EMA50 trend filter to avoid counter-trend trades, volume confirmation to ensure breakout strength, and Donchian breakouts for clear entry/exit signals.
# Designed for 20-50 trades/year by requiring multiple confluence factors: Donchian breakout, volume spike, and trend alignment.
# Works in bull markets via long breakouts and in bear markets via short breakdowns.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Donchian_Breakout_Volume_Confirmation_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20 period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 12h data ONCE for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > donch_high[i-1]  # Break above previous period's high
        breakout_down = close[i] < donch_low[i-1]  # Break below previous period's low
        
        # Volume confirmation: current volume > 1.5x average
        volume_spike = volume[i] > vol_ma[i] * 1.5
        
        # Trend filter from 12h EMA50
        above_ema = close[i] > ema_50_12h_aligned[i]
        below_ema = close[i] < ema_50_12h_aligned[i]
        
        # Entry conditions
        long_entry = breakout_up and volume_spike and above_ema
        short_entry = breakout_down and volume_spike and below_ema
        
        # Exit conditions: price returns to middle of Donchian channel
        donch_mid = (donch_high[i] + donch_low[i]) / 2
        long_exit = close[i] < donch_mid
        short_exit = close[i] > donch_mid
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
```

## Last Updated
2026-04-12 00:35

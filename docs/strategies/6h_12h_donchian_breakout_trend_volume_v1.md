# Strategy: 6h_12h_donchian_breakout_trend_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.007 | +17.1% | -20.7% | 74 | FAIL |
| ETHUSDT | 0.559 | +70.0% | -14.8% | 68 | PASS |
| SOLUSDT | 0.869 | +174.7% | -32.5% | 60 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.854 | +25.2% | -8.6% | 21 | PASS |
| SOLUSDT | 0.625 | +20.2% | -10.7% | 19 | PASS |

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
    
    # Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter + volume confirmation
    # Uses 12h EMA50 for trend filter: only take breakouts in direction of 12h trend
    # Volume confirmation: volume > 1.8 * 20-period average to filter false breakouts
    # Discrete sizing 0.25 to minimize fee churn. Target: 20-40 trades/year per symbol.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 6h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume confirmation: volume > 1.8 * 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine 12h trend
        bullish_trend = close[i] > ema50_12h_aligned[i]
        bearish_trend = close[i] < ema50_12h_aligned[i]
        
        # Entry logic: Donchian breakout with volume and trend filter
        long_entry = False
        short_entry = False
        
        # Long breakout: price breaks above Donchian high in bullish trend
        if bullish_trend:
            long_entry = (close[i] > donchian_high[i]) and volume_spike[i]
        # Short breakout: price breaks below Donchian low in bearish trend
        elif bearish_trend:
            short_entry = (close[i] < donchian_low[i]) and volume_spike[i]
        
        # Exit logic: opposite Donchian level or trend reversal
        long_exit = (bearish_trend and close[i] < donchian_low[i]) or (not bullish_trend and not bearish_trend)
        short_exit = (bullish_trend and close[i] > donchian_high[i]) or (not bullish_trend and not bearish_trend)
        
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
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_donchian_breakout_trend_volume_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-12 22:05

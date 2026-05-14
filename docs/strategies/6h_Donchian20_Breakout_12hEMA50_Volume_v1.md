# Strategy: 6h_Donchian20_Breakout_12hEMA50_Volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.370 | -2.1% | -21.9% | 105 | FAIL |
| ETHUSDT | 0.127 | +25.8% | -15.2% | 102 | PASS |
| SOLUSDT | 1.151 | +240.3% | -26.0% | 89 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.471 | +14.7% | -10.2% | 31 | PASS |
| SOLUSDT | -0.016 | +3.8% | -16.1% | 33 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h trend filter and volume confirmation.
# Long when price breaks above 20-period high AND 12h EMA50 uptrend AND volume > 1.5x 20-period median.
# Short when price breaks below 20-period low AND 12h EMA50 downtrend AND volume > 1.5x 20-period median.
# Donchian channels provide clear structure; 12h EMA50 filters for intermediate trend; volume spike confirms breakout strength.
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years).

name = "6h_Donchian20_Breakout_12hEMA50_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 20-period Donchian channels (using lookback of 20 periods, excluding current)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for EMA, Donchian, and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or 
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 12h EMA50 direction
        uptrend = curr_close > ema_50_12h_aligned[i]
        downtrend = curr_close < ema_50_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above 20-period high AND uptrend AND volume spike
            if curr_close > high_20[i] and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-period low AND downtrend AND volume spike
            elif curr_close < low_20[i] and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below 20-period low OR trend turns down
            if curr_close < low_20[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above 20-period high OR trend turns up
            if curr_close > high_20[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-01 12:47

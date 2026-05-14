# Strategy: 6h_Donchian20_1wEMA50_Trend_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.247 | +30.4% | -10.6% | 37 | KEEP |
| ETHUSDT | 0.045 | +21.9% | -12.9% | 35 | KEEP |
| SOLUSDT | 0.414 | +50.6% | -15.4% | 27 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.565 | -5.4% | -7.1% | 13 | DISCARD |
| ETHUSDT | 0.335 | +10.1% | -9.1% | 10 | KEEP |
| SOLUSDT | -0.475 | -0.7% | -14.5% | 9 | DISCARD |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly EMA50 trend filter and volume confirmation
# Uses 6h timeframe for entries with 1w HTF for trend alignment to reduce noise and false breakouts.
# Donchian(20) from previous 6h provides structure for breakouts.
# 1w EMA50 filters trades to only take breakouts in direction of weekly trend.
# Volume confirmation (2.0x 48-period average on 6h) ensures institutional participation.
# Designed for low trade frequency (~50-150 total trades over 4 years) to minimize fee drag.
# Works in bull markets via trend-aligned breakouts, in bear via avoidance of counter-trend false breakouts.
# Target: BTC/ETH/SOL with Sharpe > 0 on both train and test.

name = "6h_Donchian20_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian(20) levels from previous 6h bar (wait for 6h bar close)
    # We need to calculate Donchian on 6h data, then align to 6h timeframe
    # Since we're already on 6h timeframe, we can calculate directly
    high_6h = pd.Series(high)
    low_6h = pd.Series(low)
    donchian_high = high_6h.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_6h.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate volume confirmation (2.0x 48-period average on 6h timeframe)
    vol_ma = pd.Series(volume).rolling(window=48, min_periods=48).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian, EMA, and volume MA)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian high + price > 1w EMA50 + volume confirm
            if close[i] > donchian_high[i] and close[i] > ema_50_1w_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + price < 1w EMA50 + volume confirm
            elif close[i] < donchian_low[i] and close[i] < ema_50_1w_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Donchian low (strong reversal signal)
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian high (strong reversal signal)
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-02 19:25

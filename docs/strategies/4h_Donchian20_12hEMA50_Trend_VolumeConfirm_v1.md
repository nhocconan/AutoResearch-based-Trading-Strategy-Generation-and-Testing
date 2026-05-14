# Strategy: 4h_Donchian20_12hEMA50_Trend_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.284 | +36.0% | -15.8% | 84 | KEEP |
| ETHUSDT | 0.335 | +42.4% | -17.9% | 87 | KEEP |
| SOLUSDT | 0.581 | +87.9% | -34.3% | 77 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.697 | -12.8% | -14.1% | 40 | DISCARD |
| ETHUSDT | 0.095 | +6.7% | -9.3% | 33 | KEEP |
| SOLUSDT | -0.408 | -3.0% | -17.5% | 28 | DISCARD |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume spike confirmation
# Uses 4h timeframe for entries with 12h HTF for trend alignment to reduce false breakouts.
# Donchian(20) from previous 4h bar provides structure for breakouts.
# 12h EMA50 filters trades to only take breakouts in direction of intermediate trend.
# Volume confirmation (2.0x 96-period average on 4h) ensures institutional participation.
# Designed for low trade frequency (~75-200 total trades over 4 years) to minimize fee drag.
# Works in bull markets via trend-aligned breakouts, in bear via avoidance of counter-trend false breakouts.
# Target: BTC/ETH/SOL with Sharpe > 0 on both train and test.

name = "4h_Donchian20_12hEMA50_Trend_VolumeConfirm_v1"
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
    
    # Load 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian(20) levels from previous 4h bar (wait for 4h bar close)
    high_4h = pd.Series(high)
    low_4h = pd.Series(low)
    donchian_high = high_4h.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_4h.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate volume confirmation (2.0x 96-period average on 4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=96, min_periods=96).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian, EMA, and volume MA)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian high + price > 12h EMA50 + volume confirm
            if close[i] > donchian_high[i] and close[i] > ema_50_12h_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + price < 12h EMA50 + volume confirm
            elif close[i] < donchian_low[i] and close[i] < ema_50_12h_aligned[i] and volume_confirm[i]:
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

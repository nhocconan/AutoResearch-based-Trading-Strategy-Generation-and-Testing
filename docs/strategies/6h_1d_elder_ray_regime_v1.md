# Strategy: 6h_1d_elder_ray_regime_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.080 | +22.8% | -17.9% | 103 | PASS |
| ETHUSDT | -0.083 | +8.0% | -19.2% | 110 | FAIL |
| SOLUSDT | 0.842 | +168.2% | -30.1% | 109 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.043 | +6.1% | -6.2% | 36 | PASS |
| SOLUSDT | -0.000 | +3.6% | -15.2% | 26 | FAIL |

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
    
    # Hypothesis: 6h Elder Ray + 1d regime filter
    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    # Long when Bull Power > 0 AND 1d close > 1d SMA50 (bull regime)
    # Short when Bear Power > 0 AND 1d close < 1d SMA50 (bear regime)
    # Exit when power crosses zero or regime changes
    # Uses discrete sizing (0.25) targeting 50-150 trades over 4 years.
    # Works in bull/bear via regime filter ensuring we trade with higher timeframe trend.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA13 for Elder Ray and SMA50 for regime
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    
    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = ema13_1d - low_1d
    
    # Align 1d indicators to 6h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or 
            np.isnan(sma50_1d_aligned[i]) or np.isnan(close_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: bull regime when close > SMA50, bear regime when close < SMA50
        bull_regime = close_1d_aligned[i] > sma50_1d_aligned[i]
        bear_regime = close_1d_aligned[i] < sma50_1d_aligned[i]
        
        # Elder Ray signals
        long_signal = bull_power_1d_aligned[i] > 0 and bull_regime
        short_signal = bear_power_1d_aligned[i] > 0 and bear_regime
        
        # Exit conditions: power crosses zero or regime change
        exit_long = bull_power_1d_aligned[i] <= 0 or not bull_regime
        exit_short = bear_power_1d_aligned[i] <= 0 or not bear_regime
        
        # Entry conditions
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
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

name = "6h_1d_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-13 06:47

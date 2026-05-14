# Strategy: 6h_EMA34_1d_Donchian20_4h_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.261 | +33.7% | -12.0% | 83 | PASS |
| ETHUSDT | 0.135 | +26.6% | -20.0% | 79 | PASS |
| SOLUSDT | 0.835 | +129.0% | -24.0% | 70 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.825 | -3.6% | -9.7% | 32 | FAIL |
| ETHUSDT | 0.242 | +9.5% | -8.0% | 30 | PASS |
| SOLUSDT | -0.060 | +3.5% | -13.0% | 26 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy combining 1-day EMA trend filter (EMA34), 4-hour Donchian breakout (20-period),
# and volume confirmation. Uses EMA34 from daily timeframe to establish longer-term trend bias,
# Donchian breakout from 4h for entry timing, and volume spike to confirm momentum.
# Designed to work in both bull and bear markets by requiring alignment between daily trend
# and 4h breakout direction, reducing false signals in choppy conditions.
# Target: 15-25 trades/year per symbol with disciplined entries.
name = "6h_EMA34_1d_Donchian20_4h_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily EMA34 for trend bias
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 4-hour Donchian channels (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    donchian_high_4h = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    donchian_low_4h = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian levels to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    
    # Volume spike: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Donchian high, above daily EMA34, with volume spike
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 4h Donchian low, below daily EMA34, with volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below 4h Donchian low or below daily EMA34
            if (close[i] < donchian_low_aligned[i]) or (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above 4h Donchian high or above daily EMA34
            if (close[i] > donchian_high_aligned[i]) or (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-19 20:22

# Strategy: 6h_elder_ray_1d_trend_volume_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.082 | +15.7% | -11.7% | 695 | FAIL |
| ETHUSDT | 0.092 | +23.9% | -20.2% | 706 | PASS |
| SOLUSDT | 0.319 | +44.1% | -38.2% | 669 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.070 | +6.4% | -9.5% | 229 | PASS |
| SOLUSDT | -0.069 | +3.8% | -15.0% | 235 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Elder Ray with 1d Trend Filter and Volume Confirmation
# Hypothesis: Elder Ray (Bull Power/Bear Power) identifies trend strength.
# Combined with 1d EMA50 trend filter to ensure trades align with higher timeframe trend.
# Volume confirmation ensures moves have institutional participation.
# Works in both bull and bear markets by only taking trades in direction of 1d trend.
# Targets 15-30 trades/year with disciplined entries to avoid overtrading.

name = "6h_elder_ray_1d_trend_volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 13-period EMA for Elder Ray calculation
    ema13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    
    # Elder Ray components
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 20-period SMA for volume average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup for EMA and volume SMA
        # Skip if required data not available
        if (np.isnan(ema50_6h[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: bear power becomes positive (selling pressure gone) OR trend turns down
            if bear_power[i] > 0 or close[i] < ema50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: bull power becomes negative (buying pressure gone) OR trend turns up
            if bull_power[i] < 0 or close[i] > ema50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: bull power positive (buying pressure) + volume confirmation + uptrend
            if (bull_power[i] > 0 and 
                vol_confirm and 
                close[i] > ema50_6h[i]):
                position = 1
                signals[i] = 0.25
            # Short: bear power negative (selling pressure) + volume confirmation + downtrend
            elif (bear_power[i] < 0 and 
                  vol_confirm and 
                  close[i] < ema50_6h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-07 14:06

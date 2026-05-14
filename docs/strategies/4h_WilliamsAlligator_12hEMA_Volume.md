# Strategy: 4h_WilliamsAlligator_12hEMA_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.062 | +22.4% | -13.8% | 159 | PASS |
| ETHUSDT | 0.337 | +42.5% | -13.7% | 155 | PASS |
| SOLUSDT | 0.728 | +114.3% | -21.7% | 137 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.511 | -10.9% | -12.2% | 66 | FAIL |
| ETHUSDT | 0.749 | +20.1% | -8.0% | 51 | PASS |
| SOLUSDT | -0.182 | +1.3% | -11.9% | 42 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 12h EMA trend filter and volume confirmation
# Uses 12h EMA50 for stronger trend bias, reducing false signals in chop
# Williams Alligator identifies trend alignment: Lips>Teeth>Jaw for uptrend, Jaw>Teeth>Lips for downtrend
# Volume spike (>2x 20-period average) confirms momentum
# Target: 20-30 trades/year per symbol with disciplined entries
name = "4h_WilliamsAlligator_12hEMA_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h EMA50 for trend bias
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Williams Alligator components (SMMA = Smoothed Moving Average)
    def smoothed_moving_average(data, period):
        sma = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            sma[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                sma[i] = (sma[i-1] * (period-1) + data[i]) / period
        return sma
    
    # Calculate Alligator lines on 4h data
    jaw = smoothed_moving_average(close, 13)  # Blue line (13-period)
    teeth = smoothed_moving_average(close, 8)   # Red line (8-period)
    lips = smoothed_moving_average(close, 5)    # Green line (5-period)
    
    # Volume spike: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + above 12h EMA + volume spike
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Jaw > Teeth > Lips (bearish alignment) + below 12h EMA + volume spike
            elif (jaw[i] > teeth[i] and teeth[i] > lips[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if Alligator lines intertwine (Lips < Teeth) or price breaks below 12h EMA
            if (lips[i] < teeth[i]) or (close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if Alligator lines intertwine (Jaw < Teeth) or price breaks above 12h EMA
            if (jaw[i] < teeth[i]) or (close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-19 21:18

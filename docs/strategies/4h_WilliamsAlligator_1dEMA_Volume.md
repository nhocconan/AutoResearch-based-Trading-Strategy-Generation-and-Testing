# Strategy: 4h_WilliamsAlligator_1dEMA_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.332 | +39.5% | -10.7% | 183 | PASS |
| ETHUSDT | 0.112 | +24.8% | -15.7% | 196 | PASS |
| SOLUSDT | 0.909 | +161.8% | -26.8% | 185 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.192 | -7.7% | -10.2% | 75 | FAIL |
| ETHUSDT | 0.672 | +18.5% | -7.9% | 58 | PASS |
| SOLUSDT | -0.058 | +3.3% | -13.1% | 64 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + volume confirmation + 1d EMA trend filter
# Williams Alligator consists of three smoothed moving averages (Jaws 13, Teeth 8, Lips 5)
# In strong trends, the lines are well-separated and aligned (Jaws > Teeth > Lips for uptrend)
# In ranging markets, the lines intertwine and frequently cross
# Combined with volume confirmation and daily EMA trend filter to reduce false signals
# Target: 20-30 trades/year per symbol with disciplined entries
name = "4h_WilliamsAlligator_1dEMA_Volume"
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
    
    # Daily EMA34 for trend bias
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
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
    
    # Volume spike: volume > 1.5 * 30-period average
    volume_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + above daily EMA + volume spike
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Jaw > Teeth > Lips (bearish alignment) + below daily EMA + volume spike
            elif (jaw[i] > teeth[i] and teeth[i] > lips[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if Alligator lines intertwine (Lips < Teeth) or price breaks below daily EMA
            if (lips[i] < teeth[i]) or (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if Alligator lines intertwine (Jaw < Teeth) or price breaks above daily EMA
            if (jaw[i] < teeth[i]) or (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-19 20:48

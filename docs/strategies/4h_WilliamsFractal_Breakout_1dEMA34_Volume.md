# Strategy: 4h_WilliamsFractal_Breakout_1dEMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.312 | +33.2% | -9.4% | 121 | PASS |
| ETHUSDT | 0.175 | +28.3% | -14.1% | 92 | PASS |
| SOLUSDT | 1.045 | +111.6% | -12.5% | 66 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.820 | +12.3% | -3.6% | 36 | PASS |
| ETHUSDT | 1.038 | +18.4% | -4.8% | 38 | PASS |
| SOLUSDT | 2.057 | +40.3% | -6.1% | 25 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Fractal Breakout with 1d EMA34 Trend Filter and Volume Confirmation
# Uses Williams Fractals to identify potential reversal points, confirmed by 1d EMA34 trend direction
# Volume spike (>2.0x 20-period average) ensures institutional participation
# Works in both bull and bear markets by trading breakouts in the direction of the daily trend
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Discrete sizing 0.25 balances profit potential and fee drag

name = "4h_WilliamsFractal_Breakout_1dEMA34_Volume"
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Fractals (5-bar: 2 left, 2 right)
    # Bearish fractal: high[n] > high[n-2] and high[n] > high[n-1] and high[n] > high[n+1] and high[n] > high[n+2]
    # Bullish fractal: low[n] < low[n-2] and low[n] < low[n-1] and low[n] < low[n+1] and low[n] < low[n+2]
    bearish_fractal = np.zeros(n, dtype=bool)
    bullish_fractal = np.zeros(n, dtype=bool)
    
    for i in range(2, n-2):
        if (high[i] > high[i-2] and high[i] > high[i-1] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish_fractal[i] = True
        if (low[i] < low[i-2] and low[i] < low[i-1] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            bullish_fractal[i] = True
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bullish fractal breakout AND price > 1d EMA34 (uptrend) AND volume spike
            if (bullish_fractal[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Bearish fractal breakout AND price < 1d EMA34 (downtrend) AND volume spike
            elif (bearish_fractal[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below 1d EMA34 (trend change) OR opposite fractal with volume
            if (close[i] < ema_34_1d_aligned[i] or 
                (bearish_fractal[i] and volume_spike[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close above 1d EMA34 (trend change) OR opposite fractal with volume
            if (close[i] > ema_34_1d_aligned[i] or 
                (bullish_fractal[i] and volume_spike[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-02 07:06

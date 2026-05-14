# Strategy: 6h_fractal_breakout_1d_trend_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.189 | +30.2% | -13.1% | 70 | KEEP |
| ETHUSDT | -0.134 | +6.2% | -27.6% | 78 | DISCARD |
| SOLUSDT | 0.756 | +132.7% | -31.8% | 76 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.876 | -5.2% | -8.8% | 30 | DISCARD |
| SOLUSDT | 0.164 | +8.0% | -11.3% | 23 | KEEP |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import align_htf_to_ltf, compute_williams_fractal_levels, get_htf_data

name = "6h_fractal_breakout_1d_trend_volume_v1"
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
    
    # 1d data for trend and fractal detection
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    bearish_levels, bullish_levels = compute_williams_fractal_levels(high_1d, low_1d)

    # Fractals require two future 1d candles for confirmation.
    bearish_fractal_6h = pd.Series(
        align_htf_to_ltf(prices, df_1d, bearish_levels, additional_delay_bars=2)
    ).ffill().to_numpy()
    bullish_fractal_6h = pd.Series(
        align_htf_to_ltf(prices, df_1d, bullish_levels, additional_delay_bars=2)
    ).ffill().to_numpy()
    
    # 1d trend: 34-period EMA (responsive trend)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: 6h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(34, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_6h[i]) or np.isnan(bearish_fractal_6h[i]) or 
            np.isnan(bullish_fractal_6h[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: break below confirmed support or trend fails
            if close[i] < bullish_fractal_6h[i] or close[i] < ema_34_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: break above confirmed resistance or trend fails
            if close[i] > bearish_fractal_6h[i] or close[i] > ema_34_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter
            bullish = close[i] > ema_34_6h[i]
            bearish = close[i] < ema_34_6h[i]
            
            # Long: break above confirmed resistance with bullish trend + volume
            if (close[i] > bearish_fractal_6h[i] and bullish and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: break below confirmed support with bearish trend + volume
            elif (close[i] < bullish_fractal_6h[i] and bearish and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-08 10:56

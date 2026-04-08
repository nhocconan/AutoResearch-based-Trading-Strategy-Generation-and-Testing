# Strategy: 6h_williamsr_1d_ema_vol_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.732 | +2.2% | -17.4% | 106 | DISCARD |
| ETHUSDT | 0.458 | +39.6% | -14.0% | 88 | KEEP |
| SOLUSDT | 0.168 | +28.3% | -13.9% | 69 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.528 | +12.3% | -7.0% | 39 | KEEP |
| SOLUSDT | -1.513 | -9.6% | -12.3% | 31 | DISCARD |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d EMA trend filter with volume confirmation
# Long when Williams %R(14) crosses above -50 AND price > 1d EMA(50) AND volume > 1.5x 20-period average
# Short when Williams %R(14) crosses below -50 AND price < 1d EMA(50) AND volume > 1.5x 20-period average
# Uses 1d EMA for trend filter to avoid counter-trend trades, Williams %R for momentum timing, volume for confirmation
# Target: 50-150 total trades over 4 years (12-37/year) to stay within optimal range for 6h timeframe

name = "6h_williamsr_1d_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = williams_r.values  # Convert to numpy array
    
    # 1-day EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    
    # Calculate 50-period EMA on daily close
    daily_close_series = pd.Series(daily_close)
    daily_ema = daily_close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align daily EMA to 6h timeframe
    daily_ema_aligned = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(williams_r[i]) or np.isnan(daily_ema_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse position or trend filter fails
        if position == 1:  # long position
            # Exit: Williams %R crosses below -50 or price below daily EMA
            if (williams_r[i] < -50 and williams_r[i-1] >= -50) or close[i] < daily_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Williams %R crosses above -50 or price above daily EMA
            if (williams_r[i] > -50 and williams_r[i-1] <= -50) or close[i] > daily_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend filter and volume confirmation
            # Long: Williams %R crosses above -50 AND price > daily EMA AND volume confirmation
            if (williams_r[i] > -50 and williams_r[i-1] <= -50 and 
                close[i] > daily_ema_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -50 AND price < daily EMA AND volume confirmation
            elif (williams_r[i] < -50 and williams_r[i-1] >= -50 and 
                  close[i] < daily_ema_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals
```

## Last Updated
2026-04-07 04:13

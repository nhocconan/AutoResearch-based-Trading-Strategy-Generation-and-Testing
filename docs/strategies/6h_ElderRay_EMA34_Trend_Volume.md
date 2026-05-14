# Strategy: 6h_ElderRay_EMA34_Trend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.091 | +11.6% | -16.0% | 171 | DISCARD |
| ETHUSDT | 0.203 | +32.5% | -15.0% | 169 | KEEP |
| SOLUSDT | 1.093 | +250.2% | -28.1% | 166 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.574 | +17.8% | -9.3% | 56 | KEEP |
| SOLUSDT | -0.018 | +3.4% | -11.8% | 58 | DISCARD |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray components
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 13-period EMA for Elder Ray (using close)
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align to 6h timeframe
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Get 12h data for trend filter (EMA34)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_6h = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volume filter: current volume > 1.3 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 40  # Need Elder Ray, EMA34, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bull_power_6h[i]) or 
            np.isnan(bear_power_6h[i]) or 
            np.isnan(ema34_6h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.3 * volume_ma20[i])
        
        # Trend filter: price above/below 12h EMA34
        price_above_ema = close[i] > ema34_6h[i]
        price_below_ema = close[i] < ema34_6h[i]
        
        if position == 0:
            # Long: Bull Power positive AND price above 12h EMA34 with volume
            if (bull_power_6h[i] > 0 and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power negative AND price below 12h EMA34 with volume
            elif (bear_power_6h[i] < 0 and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power turns negative OR price crosses below 12h EMA34
            if (bull_power_6h[i] <= 0) or (close[i] < ema34_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power turns positive OR price crosses above 12h EMA34
            if (bear_power_6h[i] >= 0) or (close[i] > ema34_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_EMA34_Trend_Volume"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-23 21:46

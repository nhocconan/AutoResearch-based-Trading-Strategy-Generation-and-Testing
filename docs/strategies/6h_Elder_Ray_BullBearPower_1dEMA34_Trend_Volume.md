# Strategy: 6h_Elder_Ray_BullBearPower_1dEMA34_Trend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.097 | +14.6% | -13.6% | 152 | FAIL |
| ETHUSDT | 0.127 | +26.1% | -16.2% | 152 | PASS |
| SOLUSDT | 0.865 | +140.9% | -29.4% | 146 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.210 | +8.8% | -8.5% | 57 | PASS |
| SOLUSDT | -0.134 | +2.4% | -13.0% | 47 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power + 1d EMA34 trend filter + volume confirmation
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 AND Bear Power rising (less negative) AND close > 1d EMA34 AND volume > 1.5x 20-period average
# Short when Bear Power < 0 AND Bull Power falling (less positive) AND close < 1d EMA34 AND volume > 1.5x 20-period average
# Exit when Bull Power <= 0 (for long) or Bear Power >= 0 (for short) OR close crosses 1d EMA34
# Uses 6h primary timeframe with 1d HTF for trend filter to capture sustained moves with controlled frequency
# Discrete sizing (0.25) to limit fee drag and manage drawdown in both bull and bear markets
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# Elder Ray measures bull/bear strength relative to trend; EMA34 filters for higher-timeframe trend; volume confirms participation

name = "6h_Elder_Ray_BullBearPower_1dEMA34_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Elder Ray components on 6h data
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power: High - EMA13
    bear_power = low - ema_13   # Bear Power: Low - EMA13
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0 AND Bear Power rising (less negative than previous) 
            #              AND close > 1d EMA34 AND volume spike
            if (bull_power[i] > 0 and 
                i > 50 and bear_power[i] > bear_power[i-1] and  # Bear Power rising
                close[i] > ema_34_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power < 0 AND Bull Power falling (less positive than previous)
            #               AND close < 1d EMA34 AND volume spike
            elif (bear_power[i] < 0 and 
                  i > 50 and bull_power[i] < bull_power[i-1] and  # Bull Power falling
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 (weakening bullish strength) OR close < 1d EMA34 (trend flip)
            if bull_power[i] <= 0 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 (weakening bearish strength) OR close > 1d EMA34 (trend flip)
            if bear_power[i] >= 0 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-05 04:47

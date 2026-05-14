# Strategy: 4h_Camarilla_R4_S4_Breakout_12hEMA50_Trend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.487 | +55.3% | -18.3% | 61 | PASS |
| ETHUSDT | 0.334 | +44.5% | -15.2% | 56 | PASS |
| SOLUSDT | 1.038 | +201.0% | -27.0% | 46 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.147 | +3.2% | -8.0% | 23 | FAIL |
| ETHUSDT | 0.333 | +12.2% | -11.4% | 18 | PASS |
| SOLUSDT | 0.352 | +12.6% | -11.3% | 18 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 12h trend filter and volume confirmation
# Long when price breaks above Camarilla R4 AND 12h close > 12h EMA50 AND volume > 2.0x 20-period average
# Short when price breaks below Camarilla S4 AND 12h close < 12h EMA50 AND volume > 2.0x 20-period average
# Exit when price crosses 12h EMA50 (trend reversal)
# Uses 4h primary timeframe with 12h HTF for trend filter and Camarilla structure
# Discrete sizing (0.30) to limit fee drag and manage drawdown
# Target: 100-180 total trades over 4 years (25-45/year) based on proven Camarilla breakout performance
# Works in both bull and bear markets by following the 12h trend while using 4h for entry timing

name = "4h_Camarilla_R4_S4_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for trend filter and Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h close for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels on 12h data (based on previous bar's OHLC)
    # Camarilla levels: R4 = close + 1.5 * (high - low), S4 = close - 1.5 * (high - low)
    camarilla_r4 = df_12h['close'].values + 1.5 * (df_12h['high'].values - df_12h['low'].values)
    camarilla_s4 = df_12h['close'].values - 1.5 * (df_12h['high'].values - df_12h['low'].values)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # Volume confirmation: volume > 2.0x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R4 AND 12h close > 12h EMA50 AND volume spike
            if (close[i] > camarilla_r4_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.30
                position = 1
            # Short conditions: price breaks below Camarilla S4 AND 12h close < 12h EMA50 AND volume spike
            elif (close[i] < camarilla_s4_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses below 12h EMA50 (trend reversal)
            if close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses above 12h EMA50 (trend reversal)
            if close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
```

## Last Updated
2026-05-05 06:08

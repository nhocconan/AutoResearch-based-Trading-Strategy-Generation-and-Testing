# Strategy: 6h_Camarilla_R3S3_12hEMA34_VolumeConfirm

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.107 | +24.9% | -10.0% | 138 | PASS |
| ETHUSDT | 0.333 | +38.9% | -9.3% | 115 | PASS |
| SOLUSDT | 0.892 | +124.1% | -19.6% | 101 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.004 | -3.7% | -9.7% | 53 | FAIL |
| ETHUSDT | 0.429 | +12.2% | -7.6% | 42 | PASS |
| SOLUSDT | -0.754 | -5.8% | -15.4% | 36 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume confirmation (>2.0x 20 EMA volume)
# Uses Camarilla pivot levels from prior completed 1d bar for structure (breakout at R3/S3 = momentum)
# 12h EMA34 filter ensures we trade in direction of higher timeframe trend (avoids counter-trend whipsaws)
# Volume confirmation ensures breakout has sufficient participation (>2.0x average volume)
# Discrete sizing 0.25 balances risk and return while minimizing fee churn
# Target: 50-150 total trades over 4 years = 12-37/year for 6h timeframe
# Works in both bull (breakout continuation) and bear (breakdown continuation) markets
# Focus on BTC/ETH by requiring 12h trend alignment (avoids SOL-only bias, more robust across regimes)

name = "6h_Camarilla_R3S3_12hEMA34_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 days for pivot calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivots from prior completed 1d bar
    # Camarilla levels: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), 
    #                 S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    # where C = (H+L+C)/3 (typical price)
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    rng = high_1d - low_1d
    r4 = typical_price + (rng * 1.1 / 2.0)
    r3 = typical_price + (rng * 1.1 / 4.0)
    s3 = typical_price - (rng * 1.1 / 4.0)
    s4 = typical_price - (rng * 1.1 / 2.0)
    
    # Shift by 1 to use only prior completed 1d bar (no look-ahead)
    r4_shifted = np.roll(r4, 1)
    r3_shifted = np.roll(r3, 1)
    s3_shifted = np.roll(s3, 1)
    s4_shifted = np.roll(s4, 1)
    r4_shifted[0] = np.nan
    r3_shifted[0] = np.nan
    s3_shifted[0] = np.nan
    s4_shifted[0] = np.nan
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_shifted)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_shifted)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_shifted)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_shifted)
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:  # Need enough data for EMA calculation
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(34) trend filter from prior completed 12h bar
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_shifted = np.roll(ema_34_12h, 1)
    ema_34_12h_shifted[0] = np.nan
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 AND price > 12h EMA34 AND volume spike
            if close[i] > r3_aligned[i] and close[i] > ema_34_12h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3 AND price < 12h EMA34 AND volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema_34_12h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to S3 OR price crosses below 12h EMA34
            if close[i] < s3_aligned[i] or close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to R3 OR price crosses above 12h EMA34
            if close[i] > r3_aligned[i] or close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-04 09:37

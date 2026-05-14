# Strategy: 4h_Camarilla_R3S3_Breakout_12hEMA34_Trend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.312 | +13.1% | -11.6% | 271 | FAIL |
| ETHUSDT | 0.226 | +28.6% | -7.7% | 238 | PASS |
| SOLUSDT | 0.061 | +22.4% | -15.2% | 214 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.163 | +7.5% | -5.0% | 99 | PASS |
| SOLUSDT | -0.879 | -1.6% | -9.4% | 78 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot levels with 12h trend filter and volume spike
# Uses proven Camarilla structure from top performers. 12h EMA34 ensures trend alignment.
# Volume spike >2.0 filters false breakouts. Works in bull via R3/S3 breaks, in bear via reversals at S1/R1.
# Target: 20-40 trades/year to avoid fee drag. Discrete sizing 0.25.
name = "4h_Camarilla_R3S3_Breakout_12hEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Previous day's OHLC (using 1d data for proper daily boundaries)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's close, high, low
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Align to 4h
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    
    # Calculate Camarilla levels for current day
    range_ = prev_high_aligned - prev_low_aligned
    # Camarilla R3, S3, R1, S1
    r3 = prev_close_aligned + 1.1 * range_ * 1.1/2
    s3 = prev_close_aligned - 1.1 * range_ * 1.1/2
    r1 = prev_close_aligned + 1.1 * range_ * 1.0/12
    s1 = prev_close_aligned - 1.1 * range_ * 1.0/12
    
    # 12h EMA34 trend filter
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volume confirmation - 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 34)  # warmup period
    
    for i in range(start_idx, n):
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: break above R3 with trend alignment and volume spike
            if (close[i] > r3[i] and 
                close[i] > ema34_12h_aligned[i] and
                vol_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Short entry: break below S3 with trend alignment and volume spike
            elif (close[i] < s3[i] and 
                  close[i] < ema34_12h_aligned[i] and
                  vol_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: break below S1 (mean reversion) OR trend fails
            if close[i] < s1[i] or close[i] < ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: break above R1 (mean reversion) OR trend fails
            if close[i] > r1[i] or close[i] > ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-08 20:43

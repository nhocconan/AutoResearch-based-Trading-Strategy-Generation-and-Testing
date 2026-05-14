# Strategy: 4h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike_1.8x

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.221 | +28.3% | -8.2% | 346 | PASS |
| ETHUSDT | 0.194 | +28.4% | -6.2% | 330 | PASS |
| SOLUSDT | 0.197 | +30.7% | -17.7% | 285 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.315 | -2.9% | -6.1% | 127 | FAIL |
| ETHUSDT | 1.601 | +26.7% | -4.0% | 120 | PASS |
| SOLUSDT | 1.485 | +24.3% | -4.8% | 98 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation (1.8x)
# Long when price breaks above 4h Camarilla R3 level AND price > 12h EMA50 (uptrend) AND volume > 1.8x 20-period average
# Short when price breaks below 4h Camarilla S3 level AND price < 12h EMA50 (downtrend) AND volume > 1.8x 20-period average
# Exit when price crosses 4h Camarilla pivot point OR 12h EMA50 filter reverses
# Uses Camarilla pivot levels for structure + volume confirmation to reduce false breakouts
# 12h EMA50 provides medium-term trend filter effective in both bull and bear markets
# Designed for 75-200 total trades over 4 years (19-50/year) to minimize fee drag
# Timeframe: 4h (primary), HTF: 12h

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike_1.8x"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Camarilla pivot levels (based on previous bar's OHLC)
    # Using previous bar to avoid look-ahead
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    r3 = pivot + (range_ * 1.1 / 2)  # Camarilla R3
    s3 = pivot - (range_ * 1.1 / 2)  # Camarilla S3
    pp = pivot                       # Camarilla pivot point
    
    # Get 12h data ONCE before loop for EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(50)
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3)
    pp_aligned = align_htf_to_ltf(prices, df_4h, pp)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation on 4h (threshold: 1.8x for optimal frequency)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (1.8 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND price > EMA50 (uptrend) AND volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 AND price < EMA50 (downtrend) AND volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below pivot point OR price < EMA50 (trend weakening)
            if close[i] < pp_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above pivot point OR price > EMA50 (trend weakening)
            if close[i] > pp_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-05 15:02

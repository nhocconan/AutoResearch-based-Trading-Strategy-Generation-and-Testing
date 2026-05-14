# Strategy: 4h_Camarilla_R3S3_Breakout_VolumeSpike_ADX25

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.129 | +26.3% | -12.1% | 235 | PASS |
| ETHUSDT | 0.152 | +27.9% | -19.0% | 218 | PASS |
| SOLUSDT | 0.526 | +82.0% | -28.8% | 213 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.129 | -7.5% | -15.5% | 77 | FAIL |
| ETHUSDT | 1.153 | +30.2% | -7.9% | 70 | PASS |
| SOLUSDT | -0.190 | +0.6% | -19.7% | 76 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with volume spike confirmation and ADX trend filter
# Uses 1d Camarilla levels for institutional support/resistance
# Uses 4h ADX > 25 to ensure trending markets only (avoid chop)
# Uses 4h volume > 1.5x 20-period EMA for confirmation
# Designed for 4h timeframe targeting 20-40 trades/year with discrete sizing (0.25)
# Volume spike + ADX filter reduces false breakouts while maintaining trend alignment
# Works in bull markets (breakouts with volume) and bear markets (trend continuation signals)

name = "4h_Camarilla_R3S3_Breakout_VolumeSpike_ADX25"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 4h data for ADX and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h ADX for trend filter (period=14)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = np.abs(high_4h[1:] - low_4h[1:])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), 
                       np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    dm_minus = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), 
                        np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder's smoothing function
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            first_avg = np.nansum(values[1:period+1])
            result[period] = first_avg
            for i in range(period+1, len(values)):
                result[i] = result[i-1] - (result[i-1]/period) + values[i]
        return result
    
    tr_period = 14
    tr_smoothed = wilders_smoothing(tr, tr_period)
    dm_plus_smoothed = wilders_smoothing(dm_plus, tr_period)
    dm_minus_smoothed = wilders_smoothing(dm_minus, tr_period)
    
    # DI+ and DI-
    di_plus = np.where(tr_smoothed != 0, (dm_plus_smoothed / tr_smoothed) * 100, 0)
    di_minus = np.where(tr_smoothed != 0, (dm_minus_smoothed / tr_smoothed) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smoothing(dx, tr_period)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Calculate 4h volume EMA(20) for volume confirmation
    vol_4h = df_4h['volume'].values
    vol_series = pd.Series(vol_4h)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ema_20)
    
    # Calculate camarilla levels: R3, S3 from 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_range = high_1d - low_1d
    r3 = close_1d + 1.1 * camarilla_range / 2
    s3 = close_1d - 1.1 * camarilla_range / 2
    
    # Align camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ema_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1.5 x 20-period EMA
        volume_confirmed = volume[i] > (1.5 * vol_ema_20_aligned[i])
        
        if position == 0:
            # Long: close breaks above R3 + volume confirmation + ADX > 25 (uptrend)
            if (close[i] > r3_aligned[i] and volume_confirmed and 
                adx_4h_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: close breaks below S3 + volume confirmation + ADX > 25 (downtrend)
            elif (close[i] < s3_aligned[i] and volume_confirmed and 
                  adx_4h_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below S3 (mean reversion) OR ADX < 20 (trend weakening)
            if close[i] < s3_aligned[i] or adx_4h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above R3 (mean reversion) OR ADX < 20 (trend weakening)
            if close[i] > r3_aligned[i] or adx_4h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-04 02:46

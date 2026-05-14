# Strategy: 4H_Camarilla_R3_S3_1DTrend_VolumeBreakout_v3

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.550 | +41.9% | -6.2% | 192 | PASS |
| ETHUSDT | 0.129 | +26.0% | -9.1% | 188 | PASS |
| SOLUSDT | 0.732 | +81.2% | -11.2% | 160 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.143 | -2.8% | -6.2% | 75 | FAIL |
| ETHUSDT | 0.390 | +10.7% | -9.7% | 65 | PASS |
| SOLUSDT | 0.134 | +7.3% | -6.4% | 52 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4H_Camarilla_R3_S3_1DTrend_VolumeBreakout_v3
# Hypothesis: 4-hour timeframe strategy using daily Camarilla R3/S3 breakouts with 1-day EMA34 trend filter and volume spike confirmation.
# Targets 20-50 trades/year to minimize fee drag. Uses price channel structure with trend and volume filters.
# Works in bull markets (breakouts with trend) and bear markets (fades from extremes with trend filter).
# Added volatility filter to avoid whipsaws and reduced position size to manage drawdown.

name = "4H_Camarilla_R3_S3_1DTrend_VolumeBreakout_v3"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough data for EMA34
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels for each day
    range_1d = prev_high - prev_low
    r3 = prev_close + range_1d * 1.1 / 4
    s3 = prev_close - range_1d * 1.1 / 4
    pp = (prev_high + prev_low + prev_close) / 3  # Pivot point
    
    # Calculate 1-day EMA34 for trend filter
    ema_34 = pd.Series(prev_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align Camarilla levels, EMA, and pivot to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume filter: current volume > 2.0x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volatility filter: avoid low volatility periods (ATR < 0.5% of price)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    vol_filter = atr > 0.005 * close  # ATR > 0.5% of price
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure we have EMA34 and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(ema_34_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or np.isnan(vol_filter[i]) or not vol_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation (2.0x average volume)
        volume_filter = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above R3 + uptrend (price > EMA34) + volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_34_aligned[i] and   # Uptrend filter
                volume_filter):
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below S3 + downtrend (price < EMA34) + volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_34_aligned[i] and   # Downtrend filter
                  volume_filter):
                signals[i] = -0.20
                position = -1
        elif position != 0:
            # Exit conditions:
            # 1. Price returns to pivot point (mean reversion)
            # 2. Opposite Camarilla level break (trend exhaustion)
            at_pivot = abs(close[i] - pp_aligned[i]) < (r3_aligned[i] - pp_aligned[i]) * 0.1  # Within 10% of PP
            opposite_break = (position == 1 and close[i] < s3_aligned[i]) or \
                           (position == -1 and close[i] > r3_aligned[i])
            
            if at_pivot or opposite_break:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals
```

## Last Updated
2026-05-07 03:31

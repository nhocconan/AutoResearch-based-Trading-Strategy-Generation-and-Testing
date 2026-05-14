# Strategy: 4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.849 | +55.6% | -7.1% | 220 | PASS |
| ETHUSDT | 0.334 | +35.9% | -11.1% | 212 | PASS |
| SOLUSDT | 0.731 | +81.5% | -18.0% | 174 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.287 | -4.3% | -6.4% | 93 | FAIL |
| ETHUSDT | 1.421 | +26.3% | -7.3% | 81 | PASS |
| SOLUSDT | 0.591 | +13.5% | -7.3% | 61 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: On 4h timeframe, enter long when price breaks above Camarilla R3 with price > daily EMA34 and volume spike (>2x 20-period MA).
# Enter short when price breaks below Camarilla S3 with price < daily EMA34 and volume spike.
# Exit when price crosses back below R3 (for longs) or above S3 (for shorts).
# Uses daily timeframe for trend filter and weekly for volatility regime filter to avoid false breakouts in low volatility.
# Targets 20-40 trades/year for low fee drag and works in both bull and bear markets by fading extreme daily levels with institutional levels.

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
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
    
    # Calculate Camarilla levels from previous day (using daily data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(daily_high, 1)
    prev_low = np.roll(daily_low, 1)
    prev_close = np.roll(daily_close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla R3 and S3 levels
    camarilla_range = prev_high - prev_low
    r3 = prev_close + camarilla_range * 1.1 / 4
    s3 = prev_close - camarilla_range * 1.1 / 4
    
    # Daily EMA34 for trend filter
    daily_ema34 = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Weekly ATR for volatility regime filter (avoid low volatility breakouts)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # True Range calculation for weekly ATR
    tr1 = weekly_high - weekly_low
    tr2 = np.abs(weekly_high - np.roll(weekly_close, 1))
    tr3 = np.abs(weekly_low - np.roll(weekly_close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    weekly_atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: 20-period moving average on 4h data
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    daily_ema34_aligned = align_htf_to_ltf(prices, df_1d, daily_ema34)
    weekly_atr_aligned = align_htf_to_ltf(prices, df_1w, weekly_atr)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(daily_ema34_aligned[i]) or np.isnan(weekly_atr_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        daily_trend = daily_ema34_aligned[i]
        vol_ma_val = vol_ma[i]
        atr_val = weekly_atr_aligned[i]
        
        # Avoid trading in extremely low volatility (choppy) markets
        if atr_val < 0.5 * np.nanmedian(weekly_atr_aligned[max(0, i-50):i+1]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 with price > daily EMA34 and volume > 2x MA
            if close[i] > r3_val and close[i] > daily_trend and volume[i] > vol_ma_val * 2.0:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with price < daily EMA34 and volume > 2x MA
            elif close[i] < s3_val and close[i] < daily_trend and volume[i] > vol_ma_val * 2.0:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back below R3 (failed breakout)
            if close[i] < r3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back above S3 (failed breakout)
            if close[i] > s3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 09:05

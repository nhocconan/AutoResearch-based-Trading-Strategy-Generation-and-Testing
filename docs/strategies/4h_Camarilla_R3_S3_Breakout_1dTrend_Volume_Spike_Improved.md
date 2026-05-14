# Strategy: 4h_Camarilla_R3_S3_Breakout_1dTrend_Volume_Spike_Improved

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.404 | +33.0% | -4.5% | 135 | PASS |
| ETHUSDT | 0.199 | +28.1% | -6.6% | 114 | PASS |
| SOLUSDT | 0.277 | +35.4% | -16.7% | 100 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.095 | -5.7% | -6.3% | 55 | FAIL |
| ETHUSDT | 0.854 | +14.6% | -5.3% | 46 | PASS |
| SOLUSDT | -0.189 | +4.0% | -6.3% | 32 | FAIL |

## Code
```python
# 4h_Camarilla_R3_S3_Breakout_1dTrend_Volume_Spike_Improved
# Hypothesis: Price breaks of Camarilla R3/S3 levels from daily pivot, confirmed by daily trend and volume spikes, work across market regimes by capturing institutional breakouts. Uses tight entry conditions to limit trades (~100/year) and avoid fee drag.
# Improved with stricter volume filter (2.5x instead of 2.0) and added volatility filter to avoid choppy markets.
timeframe = "4h"
name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume_Spike_Improved"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter and context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Previous 1d bar's OHLC for Camarilla calculation
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    prev_close_1d = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels: R3, S3
    camarilla_r3 = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 2
    camarilla_s3 = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d EMA trend filter (34-period)
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volatility filter: ATR ratio to avoid choppy markets
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr > 0.5 * atr_ma  # Only trade when volatility is above half its 50-period average
    
    # Volume spike: current volume > 2.5 * 20-period average (stricter than before)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Wait for EMA and volume MA warmup
        # Skip if any critical value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(volatility_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with 1d uptrend, volume spike, and adequate volatility
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_1d_aligned[i] and 
                volume[i] > 2.5 * vol_ma[i] and
                volatility_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with 1d downtrend, volume spike, and adequate volatility
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_1d_aligned[i] and 
                  volume[i] > 2.5 * vol_ma[i] and
                  volatility_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below R3 or drops below 1d EMA
            if close[i] < camarilla_r3_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above S3 or rises above 1d EMA
            if close[i] > camarilla_s3_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-07 01:34

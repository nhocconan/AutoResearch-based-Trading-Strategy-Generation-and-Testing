# Strategy: 12h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeChopRegime_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.249 | +32.5% | -8.7% | 105 | PASS |
| ETHUSDT | 0.021 | +19.7% | -12.2% | 99 | PASS |
| SOLUSDT | 0.484 | +68.6% | -29.3% | 93 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.053 | -5.2% | -8.6% | 40 | FAIL |
| ETHUSDT | 0.488 | +14.1% | -6.7% | 32 | PASS |
| SOLUSDT | -0.205 | +1.6% | -16.0% | 32 | FAIL |

## Code
```python
#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation (>1.5x 20-bar avg volume). Uses chop regime filter (CHOP < 38.2 = trending) to avoid false breakouts in ranging markets. Discrete position sizing (0.25) minimizes fee churn. Targets 50-150 total trades over 4 years on BTC/ETH/SOL.

name = "12h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeChopRegime_v1"
timeframe = "12h"
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
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily Choppiness Index (CHOP) on 14-period for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d_arr, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d_arr, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    true_range_sum = pd.Series(true_range).rolling(window=14, min_periods=14).sum().values
    chop_1d = 100 * np.log10(atr_sum / true_range_sum) / np.log10(14)
    chop_1d = np.where(true_range_sum == 0, 50, chop_1d)  # avoid div by zero
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate Camarilla pivot levels from daily OHLC
    # R3 = close + (high - low) * 1.1 / 2
    # S3 = close - (high - low) * 1.1 / 2
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to LTF (12h)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Close breaks above R3, price > 1d EMA34, volume spike (>1.5x avg), trending regime (CHOP < 38.2)
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i] and 
                chop_1d_aligned[i] < 38.2):
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below S3, price < 1d EMA34, volume spike (>1.5x avg), trending regime (CHOP < 38.2)
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i] and 
                  chop_1d_aligned[i] < 38.2):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close position if price drops below S3 (reversal) OR chop becomes too high (choppy market)
            if (close[i] < camarilla_s3_aligned[i] or 
                chop_1d_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close position if price rises above R3 (reversal) OR chop becomes too high (choppy market)
            if (close[i] > camarilla_r3_aligned[i] or 
                chop_1d_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-13 23:17

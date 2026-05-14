# Strategy: 4h_Keltner_Breakout_RSI_Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.371 | +1.0% | -21.0% | 47 | FAIL |
| ETHUSDT | 0.067 | +22.3% | -13.3% | 40 | PASS |
| SOLUSDT | 1.176 | +230.3% | -23.1% | 42 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.024 | +5.3% | -12.2% | 19 | PASS |
| SOLUSDT | -0.791 | -9.4% | -20.2% | 17 | FAIL |

## Code
```python
#!/usr/bin/env python3
name = "4h_Keltner_Breakout_RSI_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Keltner channels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 20-period EMA for Keltner middle
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    # True Range for ATR
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = close_1d[0]
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - prev_close_1d)
    tr3 = np.abs(low_1d - prev_close_1d)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10_1d = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    # Keltner bands
    upper_keltner = ema_20_1d + 2.0 * atr_10_1d
    lower_keltner = ema_20_1d - 2.0 * atr_10_1d
    
    # Align to 4h
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Daily RSI for trend filter (avoid overbought/oversold extremes)
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Volume filter: 20-period average on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_keltner_aligned[i]) or np.isnan(lower_keltner_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(rsi_14_aligned[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold - avoid low-volume false breakouts
        volume_surge = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: Price breaks above upper Keltner band with volume
            # AND RSI not overbought (< 70) to avoid chasing tops
            if (close[i] > upper_keltner_aligned[i] and 
                volume_surge and 
                rsi_14_aligned[i] < 70):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Keltner band with volume
            # AND RSI not oversold (> 30) to avoid catching falling knives
            elif (close[i] < lower_keltner_aligned[i] and 
                  volume_surge and 
                  rsi_14_aligned[i] > 30):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to middle EMA or opposite band touch
            if position == 1:
                # Exit long: price returns to EMA20 or touches lower band
                if (close[i] < ema_20_aligned[i]) or (close[i] < lower_keltner_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to EMA20 or touches upper band
                if (close[i] > ema_20_aligned[i]) or (close[i] > upper_keltner_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-11 08:35

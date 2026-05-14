# Strategy: 4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeConfirm

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.566 | +48.4% | -9.8% | 131 | PASS |
| ETHUSDT | 0.044 | +21.3% | -12.7% | 116 | PASS |
| SOLUSDT | 0.747 | +100.4% | -17.7% | 101 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.158 | -5.6% | -10.6% | 50 | FAIL |
| ETHUSDT | 0.797 | +19.1% | -9.4% | 40 | PASS |
| SOLUSDT | -0.826 | -6.8% | -13.2% | 33 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
- Camarilla pivot levels (R3, S3) provide stronger support/resistance than R1/S1
- Only trade breakouts in direction of 1d EMA(34) trend to avoid counter-trend whipsaws
- Volume confirmation (> 2.5x 20-period average) ensures breakout has momentum
- Designed for 4h timeframe targeting 19-50 trades/year (75-200 over 4 years)
- Works in both bull and bear markets by trading with the 1d trend
- Camarilla levels adapt to volatility, providing dynamic support/resistance
"""

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
    
    # Get 1d data for Camarilla pivots and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d timeframe
    # Based on prior 1d bar's OHLC: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # where C = (H+L+C)/3 (typical price)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    camarilla_r3 = typical_price_1d + (range_1d * 1.1 / 4.0)
    camarilla_s3 = typical_price_1d - (range_1d * 1.1 / 4.0)
    
    # Align Camarilla levels to 4h timeframe (no extra delay needed for pivot levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 1d EMA(34) for trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: > 2.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine breakout conditions
        # Long: price breaks above R3 (strong resistance) with volume
        # Short: price breaks below S3 (strong support) with volume
        price_above_r3 = close[i] > r3_aligned[i]
        price_below_s3 = close[i] < s3_aligned[i]
        
        # Trend filter: price > EMA for long, price < EMA for short
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        
        if position == 0:
            # Long conditions: price breaks above R3, uptrend, volume spike
            long_signal = (price_above_r3 and 
                          uptrend and
                          volume[i] > 2.5 * vol_ma[i])
            
            # Short conditions: price breaks below S3, downtrend, volume spike
            short_signal = (price_below_s3 and 
                           downtrend and
                           volume[i] > 2.5 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.30
                position = 1
            elif short_signal:
                signals[i] = -0.30
                position = -1
        else:
            # Exit conditions: opposite level break or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls below S3 or trend turns down
                if (price_below_s3 or 
                    not uptrend):  # Trend reversal
                    exit_signal = True
            elif position == -1:
                # Exit short: price rises above R3 or trend turns up
                if (price_above_r3 or 
                    not downtrend):  # Trend reversal
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-23 17:46

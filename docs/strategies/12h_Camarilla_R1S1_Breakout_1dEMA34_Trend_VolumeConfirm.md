# Strategy: 12h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeConfirm

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.240 | +13.3% | -8.9% | 103 | FAIL |
| ETHUSDT | 0.094 | +24.3% | -7.1% | 90 | PASS |
| SOLUSDT | 0.016 | +17.5% | -20.3% | 95 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.352 | +10.3% | -5.6% | 37 | PASS |
| SOLUSDT | -1.153 | -8.0% | -15.8% | 35 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation
- Camarilla pivot levels (R1, S1) provide intraday support/resistance derived from prior day's range
- Only trade breakouts in direction of 1d EMA(34) trend to avoid counter-trend whipsaws
- Volume confirmation (> 1.8x 20-period average) ensures breakout has momentum
- Designed for 12h timeframe targeting 12-37 trades/year (50-150 over 4 years)
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
    
    # Get daily data for Camarilla pivots and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for daily timeframe
    # Based on prior day's OHLC: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C = (H+L+C)/3 (typical price)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    camarilla_r1 = typical_price_1d + (range_1d * 1.1 / 12.0)
    camarilla_s1 = typical_price_1d - (range_1d * 1.1 / 12.0)
    
    # Align Camarilla levels to 12h timeframe (no extra delay needed for pivot levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate 1d EMA(34) for trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine breakout conditions
        # Long: price breaks above R1 (resistance) with volume
        # Short: price breaks below S1 (support) with volume
        price_above_r1 = close[i] > r1_aligned[i]
        price_below_s1 = close[i] < s1_aligned[i]
        
        # Trend filter: price > EMA for long, price < EMA for short
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        
        if position == 0:
            # Long conditions: price breaks above R1, uptrend, volume spike
            long_signal = (price_above_r1 and 
                          uptrend and
                          volume[i] > 1.8 * vol_ma[i])
            
            # Short conditions: price breaks below S1, downtrend, volume spike
            short_signal = (price_below_s1 and 
                           downtrend and
                           volume[i] > 1.8 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite level break or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls below S1 or trend turns down
                if (price_below_s1 or 
                    not uptrend):  # Trend reversal
                    exit_signal = True
            elif position == -1:
                # Exit short: price rises above R1 or trend turns up
                if (price_above_r1 or 
                    not downtrend):  # Trend reversal
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-23 17:38

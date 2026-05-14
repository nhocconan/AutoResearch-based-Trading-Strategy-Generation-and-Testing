# Strategy: 4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeConfirm

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.147 | +27.0% | -12.4% | 229 | PASS |
| ETHUSDT | 0.192 | +30.5% | -14.9% | 219 | PASS |
| SOLUSDT | 0.824 | +121.3% | -22.5% | 204 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.625 | -0.8% | -6.8% | 82 | FAIL |
| ETHUSDT | 0.537 | +14.7% | -12.3% | 79 | PASS |
| SOLUSDT | 0.080 | +6.5% | -10.8% | 67 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3S3 Breakout with 12h EMA50 Trend and Volume Confirmation
- Camarilla R3/S3 levels provide high-probability reversal/breakout points from intraday extremes
- 12h EMA(50) ensures alignment with intermediate trend for multi-timeframe confirmation  
- Volume > 1.8x 20-period average confirms breakout strength and reduces false signals
- Designed for 4h timeframe targeting 20-50 trades/year (80-200 over 4 years) to minimize fee drag
- Works in bull markets via breakouts with trend, in bear markets via fade of overextended moves
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous day (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values  # shift(1) to use previous day's data
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R3, S3 levels: R3 = close + (high-low)*1.1/4, S3 = close - (high-low)*1.1/4
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (completed 1d bar only)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: > 1.8x 20-period average on 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA12h, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Camarilla breakout signals with trend filter and volume confirmation
        # Long: price breaks above Camarilla R3 + uptrend + volume spike
        # Short: price breaks below Camarilla S3 + downtrend + volume spike
        long_signal = (close[i] > camarilla_r3_aligned[i] and 
                      close[i] > ema_50_12h_aligned[i] and
                      volume[i] > 1.8 * vol_ma[i])
        
        short_signal = (close[i] < camarilla_s3_aligned[i] and 
                       close[i] < ema_50_12h_aligned[i] and
                       volume[i] > 1.8 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: trend reversal or opposite Camarilla level break
            exit_signal = False
            
            if position == 1:
                # Exit long: trend reversal or price breaks below Camarilla S3
                if (close[i] < ema_50_12h_aligned[i] or 
                    close[i] < camarilla_s3_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: trend reversal or price breaks above Camarilla R3
                if (close[i] > ema_50_12h_aligned[i] or 
                    close[i] > camarilla_r3_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-23 17:59

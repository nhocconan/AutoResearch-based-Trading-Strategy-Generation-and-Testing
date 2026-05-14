# Strategy: 4h_Camarilla_R4S4_12hTrend_VolumeConfirm

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.075 | +23.3% | -12.1% | 49 | PASS |
| ETHUSDT | 0.158 | +28.2% | -12.7% | 43 | PASS |
| SOLUSDT | 0.733 | +106.9% | -22.1% | 43 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.548 | +0.5% | -7.1% | 19 | FAIL |
| ETHUSDT | 0.356 | +11.0% | -10.6% | 15 | PASS |
| SOLUSDT | -0.237 | +0.8% | -13.5% | 15 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 12h trend filter and volume confirmation
# Long when price breaks above Camarilla R4 resistance AND 12h bullish trend (close > EMA50) AND volume > 1.8x 20-period volume EMA
# Short when price breaks below Camarilla S4 support AND 12h bearish trend (close < EMA50) AND volume > 1.8x 20-period volume EMA
# Uses Camarilla R4/S4 (stronger levels) for fewer, higher-quality breaks; 12h EMA50 for smoother trend filter; higher volume threshold (1.8x) to reduce noise.
# Target: 20-50 trades/year on 4h. Works in bull markets via longs in bullish 12h trend and bear markets via shorts in bearish 12h trend.

name = "4h_Camarilla_R4S4_12hTrend_VolumeConfirm"
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
    
    # Get 12h data for HTF trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_bullish_12h = close_12h > ema_50_12h
    trend_bearish_12h = close_12h < ema_50_12h
    
    # Align 12h trend to 4h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_12h, trend_bullish_12h.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_12h, trend_bearish_12h.astype(float))
    
    # Get prior day's OHLC for Camarilla levels (use 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R4 and S4 calculation:
    # R4 = close + 1.1 * (high - low) * 1.1666
    # S4 = close - 1.1 * (high - low) * 1.1666
    camarilla_r4_1d = close_1d + 1.1 * (high_1d - low_1d) * 1.1666
    camarilla_s4_1d = close_1d - 1.1 * (high_1d - low_1d) * 1.1666
    
    # Align prior day's Camarilla levels to 4h timeframe (wait for day to complete)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_1d)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_1d)
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.8)  # Volume at least 1.8x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R4 AND 12h bullish trend AND volume spike
            if (close[i] > camarilla_r4_aligned[i] and 
                trend_bullish_aligned[i] > 0.5 and  # 12h bullish trend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S4 AND 12h bearish trend AND volume spike
            elif (close[i] < camarilla_s4_aligned[i] and 
                  trend_bearish_aligned[i] > 0.5 and  # 12h bearish trend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Camarilla S4 OR 12h trend turns bearish
            if (close[i] < camarilla_s4_aligned[i] or 
                trend_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Camarilla R4 OR 12h trend turns bullish
            if (close[i] > camarilla_r4_aligned[i] or 
                trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-04 20:22

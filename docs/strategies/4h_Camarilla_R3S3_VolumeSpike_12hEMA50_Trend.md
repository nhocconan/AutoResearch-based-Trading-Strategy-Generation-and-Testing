# Strategy: 4h_Camarilla_R3S3_VolumeSpike_12hEMA50_Trend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.160 | +26.0% | -3.9% | 173 | PASS |
| ETHUSDT | 0.464 | +39.1% | -6.2% | 180 | PASS |
| SOLUSDT | 0.305 | +38.2% | -12.5% | 144 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.267 | -1.6% | -4.0% | 67 | FAIL |
| ETHUSDT | 1.380 | +23.3% | -3.9% | 65 | PASS |
| SOLUSDT | 0.241 | +8.4% | -6.7% | 54 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation
# Long when price breaks above Camarilla R3 AND volume > 2.0x 20-period average AND 12h EMA50 uptrend
# Short when price breaks below Camarilla S3 AND volume > 2.0x 20-period average AND 12h EMA50 downtrend
# Exit when price crosses Camarilla Pivot point OR 12h trend reverses
# Uses discrete sizing (0.25) to limit fee drag. Target: 20-35 trades/year per symbol.
# Camarilla levels provide mathematically derived support/resistance, volume spike confirms institutional participation,
# 12h EMA50 filters for higher timeframe direction to avoid counter-trend whipsaws.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.

name = "4h_Camarilla_R3S3_VolumeSpike_12hEMA50_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels on 4h data (based on previous day's OHLC)
    # We'll use the previous 4h bar's OHLC to calculate today's levels
    # For simplicity, we use rolling window of 6 bars (1 day = 6*4h) to get daily OHLC
    if len(df_4h) < 6:
        return np.zeros(n)
    
    # Calculate daily OHLC from 4h data (6 bars = 1 day)
    daily_open = df_4h['open'].rolling(window=6, min_periods=6).first().values
    daily_high = df_4h['high'].rolling(window=6, min_periods=6).max().values
    daily_low = df_4h['low'].rolling(window=6, min_periods=6).min().values
    daily_close = df_4h['close'].rolling(window=6, min_periods=6).last().values
    
    # Camarilla levels: based on previous day's range
    R3 = daily_close + (daily_high - daily_low) * 1.1 / 4
    S3 = daily_close - (daily_high - daily_low) * 1.1 / 4
    Pivot = (daily_high + daily_low + daily_close) / 3
    
    # Align Camarilla levels to prices timeframe
    R3_aligned = align_htf_to_ltf(prices, df_4h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_4h, S3)
    Pivot_aligned = align_htf_to_ltf(prices, df_4h, Pivot)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Uptrend when close > EMA50, downtrend when close < EMA50
    uptrend_12h = close_12h > ema_50_12h
    downtrend_12h = close_12h < ema_50_12h
    
    # Align 12h trend to 4h timeframe
    uptrend_12h_aligned = align_htf_to_ltf(prices, df_12h, uptrend_12h.astype(float))
    downtrend_12h_aligned = align_htf_to_ltf(prices, df_12h, downtrend_12h.astype(float))
    
    # Volume confirmation: volume > 2.0x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)  # No volume confirmation if insufficient data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or 
            np.isnan(Pivot_aligned[i]) or 
            np.isnan(uptrend_12h_aligned[i]) or 
            np.isnan(downtrend_12h_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND volume spike AND 12h EMA50 uptrend
            if (close[i] > R3_aligned[i] and 
                volume_filter[i] and 
                uptrend_12h_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND volume spike AND 12h EMA50 downtrend
            elif (close[i] < S3_aligned[i] and 
                  volume_filter[i] and 
                  downtrend_12h_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Camarilla Pivot OR 12h trend changes to downtrend
            if (close[i] < Pivot_aligned[i] or 
                downtrend_12h_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Camarilla Pivot OR 12h trend changes to uptrend
            if (close[i] > Pivot_aligned[i] or 
                uptrend_12h_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-05 01:18

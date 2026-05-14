# Strategy: 6h_Camarilla_R3_S3_1dEMA34_Trend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.177 | +28.4% | -9.3% | 231 | PASS |
| ETHUSDT | 0.189 | +30.1% | -12.8% | 226 | PASS |
| SOLUSDT | 1.099 | +176.2% | -17.0% | 215 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.939 | -3.6% | -7.9% | 87 | FAIL |
| ETHUSDT | 0.225 | +9.0% | -10.4% | 75 | PASS |
| SOLUSDT | 0.574 | +15.4% | -9.3% | 72 | PASS |

## Code
```python
# %%
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla levels provide precise support/resistance, EMA34 on 1d filters trend direction,
# and volume > 1.5x 20-period average confirms institutional participation.
# Works in bull/bear markets by requiring trend alignment. Target: 50-150 trades over 4 years.
name = "6h_Camarilla_R3_S3_1dEMA34_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d
    # Using previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    R3 = prev_close + 1.1 * (prev_high - prev_low) / 6
    S3 = prev_close - 1.1 * (prev_high - prev_low) / 6
    R4 = prev_close + 1.1 * (prev_high - prev_low) / 2
    S4 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align Camarilla levels to 6h
    R3_6h = align_htf_to_ltf(prices, df_1d, R3)
    S3_6h = align_htf_to_ltf(prices, df_1d, S3)
    R4_6h = align_htf_to_ltf(prices, df_1d, R4)
    S4_6h = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA and Camarilla calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_6h[i]) or np.isnan(R3_6h[i]) or np.isnan(S3_6h[i]) or 
            np.isnan(R4_6h[i]) or np.isnan(S4_6h[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > R3_6h[i-1]  # Break above R3
        short_breakout = close[i] < S3_6h[i-1]  # Break below S3
        
        trend_up = close[i] > ema_34_6h[i]
        trend_down = close[i] < ema_34_6h[i]
        
        if position == 0:
            # Long: bullish breakout + uptrend + volume confirmation
            if long_breakout and trend_up and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout + downtrend + volume confirmation
            elif short_breakout and trend_down and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish breakout below S3 or trend reversal
            if close[i] < S3_6h[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish breakout above R3 or trend reversal
            if close[i] > R3_6h[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# %%
```

## Last Updated
2026-05-09 12:35

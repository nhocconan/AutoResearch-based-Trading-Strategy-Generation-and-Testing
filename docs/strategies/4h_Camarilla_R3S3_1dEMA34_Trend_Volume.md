# Strategy: 4h_Camarilla_R3S3_1dEMA34_Trend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.345 | +37.8% | -9.3% | 142 | PASS |
| ETHUSDT | 0.070 | +22.4% | -13.6% | 143 | PASS |
| SOLUSDT | 1.176 | +204.4% | -21.3% | 114 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.081 | -5.3% | -9.8% | 59 | FAIL |
| ETHUSDT | 1.163 | +26.9% | -8.6% | 43 | PASS |
| SOLUSDT | -0.247 | +0.8% | -13.6% | 40 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Works in both bull and bear markets by requiring alignment with daily trend and volume confirmation.
# Uses proven Camarilla pivot levels for high-probability breakouts with low trade frequency.
name = "4h_Camarilla_R3S3_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Typical price = (H + L + C) / 3
    # Width = H - L
    # R3 = C + (H - L) * 1.1 / 2
    # S3 = C - (H - L) * 1.1 / 2
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    width = df_1d['high'] - df_1d['low']
    r3 = typical_price + width * 1.1 / 2
    s3 = typical_price - width * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    r3_4h = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3.values)
    
    # Volume filter: current volume > 2.0x 20-period average volume (strict to reduce trades)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need previous day's data for Camarilla levels
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_4h[i]) or np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Entry conditions
        bullish_breakout = close[i] > r3_4h[i]  # Break above R3
        bearish_breakout = close[i] < s3_4h[i]  # Break below S3
        trend_up = close[i] > ema_34_4h[i]
        trend_down = close[i] < ema_34_4h[i]
        
        if position == 0:
            # Long: bullish breakout + uptrend + volume confirmation
            if bullish_breakout and trend_up and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout + downtrend + volume confirmation
            elif bearish_breakout and trend_down and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish breakout or trend reversal
            if bearish_breakout or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish breakout or trend reversal
            if bullish_breakout or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-09 12:30

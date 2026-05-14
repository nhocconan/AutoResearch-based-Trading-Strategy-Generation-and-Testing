# Strategy: 12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSurge_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.098 | +17.2% | -12.0% | 70 | FAIL |
| ETHUSDT | 0.260 | +32.5% | -7.5% | 63 | PASS |
| SOLUSDT | 0.039 | +18.3% | -24.1% | 62 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.709 | +17.2% | -7.3% | 26 | PASS |
| SOLUSDT | -1.288 | -12.1% | -18.2% | 27 | FAIL |

## Code
```python
#!/usr/bin/env python3
name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSurge_v2"
timeframe = "12h"
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
    
    # 1d trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA34 trend
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    trend_up = close > ema_34_1d_aligned
    trend_down = close < ema_34_1d_aligned
    
    # Daily OHLC for Camarilla R3/S3 levels
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    camarilla_r3 = daily_close + (daily_high - daily_low) * 1.1 / 4
    camarilla_s3 = daily_close - (daily_high - daily_low) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume surge filter: current volume > 2.0x 24-period average (12-day equivalent in 12h)
    vol_ma_24 = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma_24[i] = np.mean(volume[i-24:i])
    vol_surge = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 4  # ~2 days (4*12h) to prevent overtrading
    
    start_idx = max(24, 34)  # Ensure enough data for volume MA and EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine trend direction
        trending_up = trend_up[i]
        trending_down = trend_down[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price breaks above Camarilla R3 with volume surge in 1d uptrend
            if (close[i] > r3_aligned[i] and 
                trending_up and 
                vol_surge[i]):
                signals[i] = 0.28
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below Camarilla S3 with volume surge in 1d downtrend
            elif (close[i] < s3_aligned[i] and 
                  trending_down and 
                  vol_surge[i]):
                signals[i] = -0.28
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls back below Camarilla S3 or 1d trend changes to down
            if close[i] < s3_aligned[i] or not trending_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.28
        elif position == -1:
            # Exit: Price rises back above Camarilla R3 or 1d trend changes to up
            if close[i] > r3_aligned[i] or not trending_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.28
    
    return signals

# Hypothesis: On 12h timeframe, price breaking above/below Camarilla R3/S3 levels with volume surge confirmation and 1d EMA34 trend filter captures institutional breakout momentum. Camarilla R3/S3 represent stronger support/resistance, reducing false breakouts. 1d trend filter ensures alignment with higher timeframe momentum. Volume surge filter (2.0x 24-period average) confirms institutional participation. Cooldown period prevents overtrading. Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag. Works in bull markets (breakouts above R3 in 1d uptrend) and bear markets (breakdowns below S3 in 1d downtrend). Uses discrete position sizing (0.28) to balance risk and reward while reducing fee churn.
```

## Last Updated
2026-05-07 13:03

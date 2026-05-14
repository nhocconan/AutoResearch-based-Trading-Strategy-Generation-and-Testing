# Strategy: 12h_Camarilla_R3S3_DailyEMA34_Trend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.167 | +27.0% | -6.6% | 76 | PASS |
| ETHUSDT | 0.194 | +29.5% | -8.2% | 64 | PASS |
| SOLUSDT | 0.183 | +30.4% | -22.3% | 62 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.189 | -4.2% | -8.6% | 32 | FAIL |
| ETHUSDT | 0.179 | +8.1% | -6.7% | 27 | PASS |
| SOLUSDT | -0.170 | +3.1% | -9.8% | 24 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout (R3/S3) with daily trend filter and volume spike confirmation
# Uses daily EMA34 for trend direction (only long when price > daily EMA34, short when price < daily EMA34)
# and Camarilla R3/S3 levels for precise entry/exit. Volume > 2x 20-period average confirms breakout strength.
# Daily trend filter ensures alignment with intermediate-term trend, reducing whipsaws in sideways markets.
# Camarilla levels provide statistically significant support/resistance that work in both bull and bear markets.
# Target: 20-30 trades/year to minimize fee decay while capturing high-probability moves.
# Focus on BTC/ETH as primary assets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    # R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6, R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    n_1d = len(close_1d)
    
    r3_1d = np.full(n_1d, np.nan)
    s3_1d = np.full(n_1d, np.nan)
    
    for i in range(1, n_1d):  # Start from 1 to use previous day's data
        if np.isnan(high_1d[i-1]) or np.isnan(low_1d[i-1]) or np.isnan(close_1d[i-1]):
            continue
        h_l = high_1d[i-1] - low_1d[i-1]
        c = close_1d[i-1]
        r3_1d[i] = c + h_l * 1.1 / 4
        s3_1d[i] = c - h_l * 1.1 / 4
    
    # For first day, use same day's data (will be refined next day)
    if n_1d > 0 and not (np.isnan(high_1d[0]) or np.isnan(low_1d[0]) or np.isnan(close_1d[0])):
        h_l = high_1d[0] - low_1d[0]
        c = close_1d[0]
        r3_1d[0] = c + h_l * 1.1 / 4
        s3_1d[0] = c - h_l * 1.1 / 4
    
    # Align daily Camarilla levels to 12h
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # 20-period average volume for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(vol_period, 1)  # Need at least 1 day of data
    
    for i in range(start_idx, n):
        if (np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Determine trend from daily EMA34
        uptrend = price > ema_34_1d_aligned[i]
        downtrend = price < ema_34_1d_aligned[i]
        
        # Breakout conditions at Camarilla R3/S3
        breakout_up = price > r3_1d_aligned[i]
        breakdown_down = price < s3_1d_aligned[i]
        
        # Volume confirmation: spike > 2x average
        volume_confirmation = vol_ratio > 2.0
        
        if position == 0:
            # Long: bullish breakout at R3 with uptrend and volume
            if uptrend and breakout_up and volume_confirmation:
                signals[i] = size
                position = 1
            # Short: bearish breakdown at S3 with downtrend and volume
            elif downtrend and breakdown_down and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns to S3 or breaks below EMA34
            if price < s3_1d_aligned[i] or price < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price returns to R3 or breaks above EMA34
            if price > r3_1d_aligned[i] or price > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3S3_DailyEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-27 14:09

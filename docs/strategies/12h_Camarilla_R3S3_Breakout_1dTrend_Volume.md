# Strategy: 12h_Camarilla_R3S3_Breakout_1dTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.145 | +26.1% | -6.6% | 70 | KEEP |
| ETHUSDT | 0.064 | +22.8% | -8.5% | 61 | KEEP |
| SOLUSDT | 0.177 | +29.9% | -22.3% | 60 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.627 | +0.8% | -7.1% | 24 | DISCARD |
| ETHUSDT | 0.337 | +10.3% | -6.1% | 25 | KEEP |
| SOLUSDT | -0.119 | +3.9% | -9.2% | 22 | DISCARD |

## Code
```python
#!/usr/bin/env python3
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
    
    # Get 1d data for Camarilla pivot and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    # Using previous day's high/low/close
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d EMA trend filter (34-period)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: volume > 2.0 x 24-period average (to reduce trades)
    vol_ma_24 = np.full(n, np.nan)
    for i in range(23, n):
        vol_ma_24[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Camarilla (1 day), EMA (34), volume MA (24)
    start_idx = max(1, 34, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_24[i]
        
        # Volume filter: significant volume spike (higher threshold = fewer trades)
        vol_filter = vol_now > 2.0 * vol_avg
        
        # Trend filter from 1d EMA
        bullish_trend = price > ema_34_aligned[i]
        bearish_trend = price < ema_34_aligned[i]
        
        camarilla_r3 = camarilla_r3_aligned[i]
        camarilla_s3 = camarilla_s3_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3 + volume + bullish 1d trend
            if price > camarilla_r3 and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: price breaks below S3 + volume + bearish 1d trend
            elif price < camarilla_s3 and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below S3 or trend turns bearish
            if price < camarilla_s3 or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above R3 or trend turns bullish
            if price > camarilla_r3 or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-05-13 02:15

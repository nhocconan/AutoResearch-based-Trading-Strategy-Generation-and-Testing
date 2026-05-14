# Strategy: 6h_Camarilla_R3S3_12hEMA50_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.312 | +31.0% | -5.1% | 205 | PASS |
| ETHUSDT | 0.307 | +32.7% | -7.7% | 175 | PASS |
| SOLUSDT | 0.448 | +49.2% | -11.5% | 143 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.617 | -4.4% | -6.5% | 81 | FAIL |
| ETHUSDT | 1.488 | +23.1% | -4.5% | 65 | PASS |
| SOLUSDT | -0.136 | +4.5% | -6.4% | 54 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation
# Long when price breaks above 6h Camarilla R3 level AND 12h close > 12h EMA50 AND volume > 2.0 * 20-bar average volume
# Short when price breaks below 6h Camarilla S3 level AND 12h close < 12h EMA50 AND volume > 2.0 * 20-bar average volume
# Exit when price retests the 6h Camarilla pivot point (PP)
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 75-150 total trades over 4 years (19-37/year) for 6h timeframe
# Camarilla levels provide precise intraday support/resistance structure
# 12h EMA50 filters for higher timeframe trend alignment
# Volume spike confirmation reduces false breakouts during low participation
# Works in both bull and bear markets by following the 12h trend

name = "6h_Camarilla_R3S3_12hEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 6h Camarilla levels and 12h EMA50 ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_6h) < 1 or len(df_12h) < 50:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    close_12h = df_12h['close'].values
    
    # Calculate 6h Camarilla levels (based on previous bar's OHLC)
    # Camarilla: PP = (H+L+C)/3, R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    prev_high = np.roll(high_6h, 1)
    prev_low = np.roll(low_6h, 1)
    prev_close = np.roll(close_6h, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_pp = (prev_high + prev_low + prev_close) / 3.0
    camarilla_r3 = prev_close + ((prev_high - prev_low) * 1.1 / 2.0)
    camarilla_s3 = prev_close - ((prev_high - prev_low) * 1.1 / 2.0)
    
    # Calculate 12h EMA50 trend filter
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h timeframe (wait for completed bars)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_6h, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_s3)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: volume > 2.0 * 20-bar average volume (spike filter)
    volume_series = pd.Series(volume)
    avg_volume_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > R3 level AND uptrend AND volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema50_12h_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < S3 level AND downtrend AND volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema50_12h_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests pivot point from above
            if close[i] <= camarilla_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests pivot point from below
            if close[i] >= camarilla_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-06 13:24

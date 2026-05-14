# Strategy: 4h_camarilla_breakout_v5

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.353 | +31.9% | -4.9% | 257 | PASS |
| ETHUSDT | 0.350 | +33.3% | -7.5% | 229 | PASS |
| SOLUSDT | 0.167 | +28.2% | -13.0% | 187 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.444 | -7.1% | -7.6% | 96 | FAIL |
| ETHUSDT | -0.643 | +0.5% | -7.6% | 84 | FAIL |
| SOLUSDT | 0.622 | +11.6% | -4.4% | 67 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_camarilla_breakout_v5
# Hypothesis: 4h strategy using daily Camarilla pivot levels with strict volume, trend, and chop regime filters.
# Long when price breaks above daily R4 with volume > 2.0x 20-period average, price > 4h EMA50, and chop < 61.8 (trending).
# Short when price breaks below daily S4 with volume > 2.0x 20-period average, price < 4h EMA50, and chop < 61.8 (trending).
# Exit when price closes back inside daily R3/S3 levels.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 15-30 trades/year (60-120 total over 4 years) on BTC/ETH/SOL to avoid overtrading and fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_breakout_v5"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 4h EMA50 for trend filter
    close_s = pd.Series(close)
    ema50 = close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Choppiness Index regime filter (14-period)
    atr_period = 14
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr_series = pd.Series(tr)
    atr_series = tr_series.rolling(window=atr_period, min_periods=atr_period).mean()
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    highest_high = high_series.rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = low_series.rolling(window=atr_period, min_periods=atr_period).min().values
    atr_sum = tr_series.rolling(window=atr_period, min_periods=atr_period).sum().values
    chop = 100 * np.log10(atr_sum / np.log10(atr_period) / (highest_high - lowest_low))
    
    # Get daily data for pivot levels (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    pivot_1d = typical_price_1d
    r1_1d = close_1d + (range_1d * 1.1 / 12)
    s1_1d = close_1d - (range_1d * 1.1 / 12)
    r2_1d = close_1d + (range_1d * 1.1 / 6)
    s2_1d = close_1d - (range_1d * 1.1 / 6)
    r3_1d = close_1d + (range_1d * 1.1 / 4)
    s3_1d = close_1d - (range_1d * 1.1 / 4)
    r4_1d = close_1d + (range_1d * 1.1 / 2)
    s4_1d = close_1d - (range_1d * 1.1 / 2)
    
    # Align all levels to 4h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or 
            np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(ema50[i]) or np.isnan(chop[i]) or
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average (stricter)
        volume_confirmed = volume[i] > 2.0 * volume_ma[i]
        # Trend filter: price > EMA50 for long, price < EMA50 for short
        # Regime filter: chop < 61.8 indicates trending market (good for breakouts)
        trending_market = chop[i] < 61.8
        
        if position == 1:  # Long position
            # Exit: Price closes back below daily R3 (take profit) or below daily S4 (stop)
            if close[i] < r3_1d_aligned[i] or close[i] < s4_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price closes back above daily S3 (take profit) or above daily R4 (stop)
            if close[i] > s3_1d_aligned[i] or close[i] > r4_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for breakout with volume, trend, and regime confirmation
            bullish_breakout = (close[i] > r4_1d_aligned[i]) and volume_confirmed and (close[i] > ema50[i]) and trending_market
            bearish_breakout = (close[i] < s4_1d_aligned[i]) and volume_confirmed and (close[i] < ema50[i]) and trending_market
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-09 01:18

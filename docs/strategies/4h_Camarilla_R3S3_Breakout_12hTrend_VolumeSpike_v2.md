# Strategy: 4h_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.292 | +14.0% | -5.7% | 111 | FAIL |
| ETHUSDT | 0.073 | +23.4% | -6.6% | 93 | PASS |
| SOLUSDT | 0.184 | +29.8% | -15.2% | 94 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.037 | +16.8% | -5.3% | 40 | PASS |
| SOLUSDT | -0.021 | +5.6% | -8.1% | 30 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike_v2"
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
    
    # Get 12h data once for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h EMA50 trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_12h = (close_12h > ema50_12h).astype(float)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # Get 1d data once for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation (R3/S3 levels)
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    prev_close[0] = df_1d['close'].values[0]
    
    # Camarilla pivot levels calculation (R3 and S3)
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    r3 = pivot + (range_val * 1.1 / 2)  # R3 level
    s3 = pivot - (range_val * 1.1 / 2)  # S3 level
    
    # Align Camarilla levels to 4h timeframe
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike detection: current volume > 2.5 * 20-period average (more stringent)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.5)
    
    # Price distance filter: require breakout to be at least 0.5% above/below level
    price_above_r3 = close > r3_4h * 1.005
    price_below_s3 = close < s3_4h * 0.995
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or np.isnan(trend_12h_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above R3 with volume spike and 12h uptrend
            long_cond = (price_above_r3[i] and vol_spike[i] and trend_12h_aligned[i] > 0.5)
            
            # Short entry: price breaks below S3 with volume spike and 12h downtrend
            short_cond = (price_below_s3[i] and vol_spike[i] and trend_12h_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverses back below R3 (mean reversion)
            if close[i] < r3_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverses back above S3 (mean reversion)
            if close[i] > s3_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakout with stricter volume confirmation (2.5x 20-period MA) and price distance filter (0.5%).
# Uses 12h EMA50 trend filter for multi-timeframe alignment.
# Reduced position size to 0.25 to lower drawdown and tighter exit conditions (reverse at pivot level) to increase win rate.
# Target: 15-30 trades/year to avoid fee drag while maintaining edge in both bull and bear markets.
```

## Last Updated
2026-05-08 10:27

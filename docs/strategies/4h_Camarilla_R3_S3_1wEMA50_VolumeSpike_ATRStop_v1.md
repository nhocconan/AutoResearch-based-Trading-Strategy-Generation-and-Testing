# Strategy: 4h_Camarilla_R3_S3_1wEMA50_VolumeSpike_ATRStop_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.176 | +28.6% | -11.9% | 170 | KEEP |
| ETHUSDT | -0.319 | +0.3% | -20.2% | 164 | DISCARD |
| SOLUSDT | 1.157 | +199.6% | -17.7% | 143 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.231 | +7.8% | -3.4% | 42 | KEEP |
| SOLUSDT | 0.925 | +17.0% | -4.3% | 28 | KEEP |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1w Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume spike confirmation.
# Long when: price breaks above 1w Camarilla R3 level AND close > 1d EMA50 AND volume > 2.0x 24-bar average
# Short when: price breaks below 1w Camarilla S3 level AND close < 1d EMA50 AND volume > 2.0x 24-bar average
# Exit via ATR(24) trailing stop: long exit when price < highest_high_since_entry - 2.5 * ATR
#                      short exit when price > lowest_low_since_entry + 2.5 * ATR
# Uses 1w Camarilla for structure (wider, more significant levels), 1d EMA50 for trend alignment, volume spike for confirmation
# Discrete sizing 0.28 balances return and fee drag. Target: 75-200 total trades over 4 years = 19-50/year.

name = "4h_Camarilla_R3_S3_1wEMA50_VolumeSpike_ATRStop_v1"
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
    
    # Calculate 1w Camarilla pivots (based on previous 1w bar)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels for each 1w bar (using previous bar's OHLC)
    camarilla_r3 = np.zeros(len(close_1w))
    camarilla_s3 = np.zeros(len(close_1w))
    camarilla_r3[0] = np.nan
    camarilla_s3[0] = np.nan
    
    for i in range(1, len(close_1w)):
        # Camarilla formulas based on previous 1w bar
        high_prev = high_1w[i-1]
        low_prev = low_1w[i-1]
        close_prev = close_1w[i-1]
        range_prev = high_prev - low_prev
        
        camarilla_r3[i] = close_prev + range_prev * 1.1 / 4  # R3 level
        camarilla_s3[i] = close_prev - range_prev * 1.1 / 4  # S3 level
    
    # Align 1w Camarilla levels to 4h timeframe (completed 1w bar only)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h ATR(24) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=24, adjust=False, min_periods=24).mean().values
    
    # Volume confirmation (2.0x 24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup (need enough for ATR, Camarilla, EMA calculations)
    start_idx = 24 + 50 + 5  # ATR(24) + EMA50 warmup + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above 1w Camarilla R3 with volume spike AND bullish trend (close > 1d EMA50)
            if close[i] > camarilla_r3_aligned[i] and volume_spike[i] and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.28
                position = 1
                entry_bar = i
                highest_since_entry = high[i]
            # Short entry: price breaks below 1w Camarilla S3 with volume spike AND bearish trend (close < 1d EMA50)
            elif close[i] < camarilla_s3_aligned[i] and volume_spike[i] and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.28
                position = -1
                entry_bar = i
                lowest_since_entry = low[i]
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, high[i])
            # ATR trailing stop: exit when price < highest_high_since_entry - 2.5 * ATR
            if close[i] < highest_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, low[i])
            # ATR trailing stop: exit when price > lowest_low_since_entry + 2.5 * ATR
            if close[i] > lowest_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals
```

## Last Updated
2026-05-03 03:29

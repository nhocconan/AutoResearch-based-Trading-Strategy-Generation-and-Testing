# Strategy: 6h_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeSpike_ATRTrail_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.026 | +21.4% | -9.5% | 137 | PASS |
| ETHUSDT | 0.439 | +44.6% | -15.1% | 118 | PASS |
| SOLUSDT | 0.885 | +119.1% | -16.9% | 110 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.611 | +0.3% | -7.2% | 54 | FAIL |
| ETHUSDT | 1.657 | +34.9% | -6.1% | 44 | PASS |
| SOLUSDT | 0.428 | +11.8% | -10.7% | 38 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above R3 with uptrend (price > 1d EMA50) and volume spike.
# Short when price breaks below S3 with downtrend (price < 1d EMA50) and volume spike.
# Uses ATR trailing stop (2.0x) for risk management.
# Targets 50-150 trades over 4 years (12-37/year) with discrete position sizing (0.25).
# Works in both bull/bear markets by requiring 1d EMA50 trend alignment.

name = "6h_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeSpike_ATRTrail_v1"
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
    
    # Load 1d data ONCE before loop for EMA50 trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla pivot levels from 1d OHLC
    # Camarilla: R4 = close + ((high-low)*1.1/2), R3 = close + ((high-low)*1.1/4), etc.
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    rng = high_1d - low_1d
    camarilla_r3 = close_1d_arr + (rng * 1.1 / 4)
    camarilla_s3 = close_1d_arr - (rng * 1.1 / 4)
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # ATR for trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = 50  # warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if Camarilla levels not available
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            continue
        
        # Regime filter: price above/below 1d EMA50 determines trend direction
        is_uptrend = close[i] > ema_50_aligned[i]
        is_downtrend = close[i] < ema_50_aligned[i]
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            if is_uptrend and curr_close > r3_aligned[i] and curr_volume_spike:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_close
            elif is_downtrend and curr_close < s3_aligned[i] and curr_volume_spike:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Update highest high since entry
            if curr_high > highest_since_entry:
                highest_since_entry = curr_high
            
            # Trailing stop: 2.0 * ATR below highest since entry
            if curr_close < highest_since_entry - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_low < lowest_since_entry:
                lowest_since_entry = curr_low
            
            # Trailing stop: 2.0 * ATR above lowest since entry
            if curr_close > lowest_since_entry + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-30 14:00

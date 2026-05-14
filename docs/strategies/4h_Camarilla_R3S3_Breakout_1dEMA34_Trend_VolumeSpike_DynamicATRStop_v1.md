# Strategy: 4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_DynamicATRStop_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.127 | +25.6% | -9.4% | 406 | PASS |
| ETHUSDT | 0.188 | +29.2% | -10.4% | 381 | PASS |
| SOLUSDT | 0.491 | +60.4% | -22.1% | 319 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.661 | +0.6% | -5.6% | 158 | FAIL |
| ETHUSDT | 0.749 | +16.2% | -6.7% | 146 | PASS |
| SOLUSDT | 0.313 | +9.8% | -7.3% | 120 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_DynamicATRStop_v1
Hypothesis: Trade Camarilla R3/S3 breakouts on 4h with 1d EMA34 trend filter and volume confirmation (2.0x median). Only trade in direction of 1d EMA34 trend to reduce whipsaws. Uses ATR trailing stop (1.5x ATR, dynamic based on volatility regime). Target: 15-25 trades/year on 4h. Works in bull/bear by adapting to trend and volatility regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels from previous 1d OHLC
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    camarilla_r3 = prev_close_1d + 3.000/6 * (prev_high_1d - prev_low_1d)
    camarilla_s3 = prev_close_1d - 3.000/6 * (prev_high_1d - prev_low_1d)
    
    # Align HTF indicators to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: 2.0x median volume (20-period) for signal
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # ATR(14) for volatility-based stops (4h ATR)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volatility regime filter: ATR ratio (current ATR / 50-period ATR) < 1.2 = low vol regime (favor mean reversion)
    atr_50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr_ratio = atr / np.where(atr_50 > 0, atr_50, np.nan)
    low_vol_regime = atr_ratio < 1.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of EMA(34) 1d, volume median (20), ATR (14) 4h, ATR 50
    start_idx = max(34, 20, 14, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_median[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(low_vol_regime[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_34_1d_val = ema_34_1d_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        atr_val = atr[i]
        low_vol = low_vol_regime[i]
        
        # Trend filter: price > EMA34 (uptrend) or < EMA34 (downtrend)
        uptrend = close_val > ema_34_1d_val
        downtrend = close_val < ema_34_1d_val
        
        if position == 0:
            # Long: break above R3 with volume spike, and uptrend (or low vol regime allows mean reversion)
            long_signal = (close_val > camarilla_r3_aligned[i]) and \
                          (volume_val > 2.0 * vol_median_val) and \
                          (uptrend or low_vol)
            
            # Short: break below S3 with volume spike, and downtrend (or low vol regime allows mean reversion)
            short_signal = (close_val < camarilla_s3_aligned[i]) and \
                           (volume_val > 2.0 * vol_median_val) and \
                           (downtrend or low_vol)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, high_val)
            # Dynamic ATR trailing stop: tighter stop in low vol, wider in high vol
            atr_multiplier = 1.0 if low_vol else 2.0
            if close_val < highest_since_entry - atr_multiplier * atr_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, low_val)
            # Dynamic ATR trailing stop: tighter stop in low vol, wider in high vol
            atr_multiplier = 1.0 if low_vol else 2.0
            if close_val > lowest_since_entry + atr_multiplier * atr_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_DynamicATRStop_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-26 02:25

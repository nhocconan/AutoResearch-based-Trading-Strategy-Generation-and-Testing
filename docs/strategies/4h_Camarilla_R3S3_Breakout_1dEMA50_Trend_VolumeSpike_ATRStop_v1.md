# Strategy: 4h_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeSpike_ATRStop_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.215 | +30.8% | -11.4% | 218 | PASS |
| ETHUSDT | 0.098 | +24.2% | -15.2% | 210 | PASS |
| SOLUSDT | 0.774 | +116.4% | -32.8% | 183 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.437 | +10.9% | -5.6% | 73 | PASS |
| ETHUSDT | 1.251 | +29.8% | -9.3% | 74 | PASS |
| SOLUSDT | 1.140 | +28.4% | -10.4% | 64 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeSpike_ATRStop_v1
Hypothesis: Camarilla R3/S3 breakout (stronger levels) on 4h with 1d EMA50 trend filter and volume spike (>2.0x median). Targets institutional pivot levels with strong volume confirmation in trending markets. Uses ATR trailing stop (2.5x) for risk management. Designed for BTC/ETH with strict entry conditions (~20-35 trades/year) to avoid overtrading. Works in bull/bear by only trading with 1d trend direction.
"""

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
    
    # Get 1d data for HTF trend (EMA50)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels from previous 4h bar (HLC of prior 4h)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    cam_high = pd.Series(df_4h['high'].values).shift(1).values
    cam_low = pd.Series(df_4h['low'].values).shift(1).values
    cam_close = pd.Series(df_4h['close'].values).shift(1).values
    
    # Camarilla R3, S3 levels (stronger breakout levels)
    R3 = cam_close + (cam_high - cam_low) * 1.1 / 4
    S3 = cam_close - (cam_high - cam_low) * 1.1 / 4
    
    # Volume spike filter: volume > 2.0x median volume (30-period) for high conviction
    vol_median = pd.Series(volume).rolling(window=30, min_periods=30).median().values
    
    # ATR(14) for volatility-based stops
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    R3_aligned = align_htf_to_ltf(prices, df_4h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_4h, S3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of EMA(50) 1d, Camarilla (need 2 bars for shift), volume median (30), ATR (14)
    start_idx = max(50, 2, 30, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(R3_aligned[i]) or
            np.isnan(S3_aligned[i]) or
            np.isnan(vol_median[i]) or
            np.isnan(atr[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        atr_val = atr[i]
        r3_val = R3_aligned[i]
        s3_val = S3_aligned[i]
        
        # Trend filter: price > EMA50 (uptrend) or < EMA50 (downtrend)
        uptrend = close_val > ema_50_1d_val
        downtrend = close_val < ema_50_1d_val
        
        # Volume spike filter: only trade in high-volume environments
        volume_spike = volume_val > 2.0 * vol_median_val
        
        if position == 0:
            # Long: break above R3 with volume spike, and uptrend
            long_signal = (close_val > r3_val) and \
                          volume_spike and \
                          uptrend
            
            # Short: break below S3 with volume spike, and downtrend
            short_signal = (close_val < s3_val) and \
                           volume_spike and \
                           downtrend
            
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
            # ATR trailing stop (wider stop to reduce whipsaw)
            if close_val < highest_since_entry - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, low_val)
            # ATR trailing stop (wider stop to reduce whipsaw)
            if close_val > lowest_since_entry + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-26 03:14

# Strategy: 6h_Camarilla_R3S3_Breakout_1dTrend_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.526 | +45.0% | -7.6% | 144 | PASS |
| ETHUSDT | 0.280 | +34.9% | -15.5% | 131 | PASS |
| SOLUSDT | 0.980 | +140.1% | -18.6% | 116 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.878 | -2.5% | -8.3% | 60 | FAIL |
| ETHUSDT | 0.494 | +13.3% | -9.6% | 50 | PASS |
| SOLUSDT | -0.102 | +3.9% | -13.5% | 41 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_VolumeConfirm_v1
Hypothesis: Trade Camarilla R3/S3 breakouts on 6h with 1d EMA34 trend filter and volume confirmation. In bullish 1d trend (close > EMA34), buy when price breaks above R3; in bearish 1d trend (close < EMA34), sell when price breaks below S3. Volume spike (2.0x 24-bar avg) confirms participation. Uses discrete position sizing (0.25) to minimize fee drag and target ~15-30 trades/year. Designed to work in both bull and bear markets by following the higher timeframe trend.
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
    
    # Get 1d data for HTF trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Camarilla R3 and S3 levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3 and S3 levels
    # R3 = close + 1.1*(high - low)/4
    # S3 = close - 1.1*(high - low)/4
    camarilla_range = high_1d - low_1d
    r3 = close_1d + 1.1 * camarilla_range / 4
    s3 = close_1d - 1.1 * camarilla_range / 4
    
    # Align Camarilla levels to 6h timeframe (1-day lagged)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3, additional_delay_bars=1)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3, additional_delay_bars=1)
    
    # Volume confirmation: 2.0x 24-bar average volume (48h = 2 days on 6h)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 and volume MA
    start_idx = max(34, 24)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend: price above/below EMA34
        trend_bullish = close[i] > ema_34_aligned[i]
        trend_bearish = close[i] < ema_34_aligned[i]
        
        if position == 0:
            # Look for breakout signals with volume confirmation and trend alignment
            long_signal = close[i] > r3_aligned[i] and volume_spike[i] and trend_bullish
            short_signal = close[i] < s3_aligned[i] and volume_spike[i] and trend_bearish
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price breaks below S3 or trend reverses
            exit_signal = close[i] < s3_aligned[i] or not trend_bullish
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price breaks above R3 or trend reverses
            exit_signal = close[i] > r3_aligned[i] or not trend_bearish
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-25 17:36

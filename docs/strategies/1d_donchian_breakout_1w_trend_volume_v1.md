# Strategy: 1d_donchian_breakout_1w_trend_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.284 | +12.4% | -9.2% | 16 | FAIL |
| ETHUSDT | -0.367 | +5.7% | -16.7% | 13 | FAIL |
| SOLUSDT | 1.042 | +132.1% | -12.7% | 13 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.606 | +15.1% | -10.9% | 12 | PASS |

## Code
```python
#!/usr/bin/env python3
# 1d_donchian_breakout_1w_trend_volume_v1
# Hypothesis: Breakout of weekly Donchian(20) channels with volume confirmation and daily trend filter.
# Long when price breaks above 20-week high with volume > 1.5x average and daily close > daily open.
# Short when price breaks below 20-week low with volume > 1.5x average and daily close < daily open.
# Exit when price returns to weekly midline or opposite signal.
# Designed to work in both bull and bear markets by capturing breakouts with trend confirmation.
# Target: 15-25 trades/year to minimize fee decay while capturing strong directional moves.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get weekly data for Donchian channels (calculate once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 20-week Donchian channels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 20-week high and low
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Weekly midline (average of high and low)
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align weekly Donchian levels to daily chart
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    # Get daily data for trend filter (calculate once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Daily trend: close > open = uptrend, close < open = downtrend
    open_1d = df_1d['open'].values
    close_1d = df_1d['close'].values
    daily_uptrend = close_1d > open_1d
    daily_downtrend = close_1d < open_1d
    
    # Align daily trend to daily chart (1:1 mapping but using the helper for consistency)
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    # Volume confirmation: 20-day average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(donchian_mid_aligned[i]) or np.isnan(daily_uptrend_aligned[i]) or \
           np.isnan(daily_downtrend_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to weekly midline or opposite signal
            if close[i] <= donchian_mid_aligned[i] or \
               (close[i] >= donchian_low_aligned[i] and volume[i] > 1.5 * avg_volume[i] and daily_downtrend_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to weekly midline or opposite signal
            if close[i] >= donchian_mid_aligned[i] or \
               (close[i] <= donchian_high_aligned[i] and volume[i] > 1.5 * avg_volume[i] and daily_uptrend_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Long entry: price breaks above 20-week high with volume and daily uptrend
            if close[i] > donchian_high_aligned[i] and volume_ok and daily_uptrend_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 20-week low with volume and daily downtrend
            elif close[i] < donchian_low_aligned[i] and volume_ok and daily_downtrend_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-08 15:29

# Strategy: 6h_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.175 | +26.9% | -7.4% | 180 | PASS |
| ETHUSDT | 0.091 | +24.1% | -9.1% | 156 | PASS |
| SOLUSDT | 0.417 | +45.8% | -14.8% | 121 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.866 | -0.3% | -6.6% | 73 | FAIL |
| ETHUSDT | 1.314 | +24.1% | -5.9% | 66 | PASS |
| SOLUSDT | -1.054 | -6.3% | -13.3% | 62 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike_v2
Hypothesis: On 6h timeframe, trade Camarilla R1/S1 breakouts with 1w EMA50 trend filter and volume spike confirmation. Weekly trend filter provides stronger directional bias than daily, reducing false breakouts in choppy markets. Volume spike confirms institutional participation. Designed for 50-150 total trades over 4 years (12-37/year) with discrete sizing (0.25) to minimize fee drag. Works in bull/bear markets via weekly trend filter.
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
    
    # Get 1w data for EMA(50) trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1d data for Camarilla pivot calculation (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: based on previous day's range
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    
    # Typical price for pivot calculation
    typical_price = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r3 = typical_price + range_hl * 1.1 / 2.0
    s3 = typical_price - range_hl * 1.1 / 2.0
    r4 = typical_price + range_hl * 1.1
    s4 = typical_price - range_hl * 1.1
    
    # Volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Align HTF indicators to 6h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of EMA(50) 1w, volume MA (20)
    start_idx = max(50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or
            np.isnan(s4_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_1w_val = ema_50_1w_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol_spike = volume_spike[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        r4_val = r4_aligned[i]
        s4_val = s4_aligned[i]
        
        # Trend filter: price > EMA50 (uptrend) or < EMA50 (downtrend)
        uptrend = close_val > ema_50_1w_val
        downtrend = close_val < ema_50_1w_val
        
        if position == 0:
            # Long: break above R3 with uptrend and volume spike (continuation)
            # OR break above R4 with volume spike (strong breakout)
            long_signal = ((high_val > r3_val and uptrend) or (high_val > r4_val)) and vol_spike
            
            # Short: break below S3 with downtrend and volume spike (continuation)
            # OR break below S4 with volume spike (strong breakout)
            short_signal = ((low_val < s3_val and downtrend) or (low_val < s4_val)) and vol_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: trend reversal or price reaches S3 (mean reversion target)
            if close_val < ema_50_1w_val or low_val <= s3_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend reversal or price reaches R3 (mean reversion target)
            if close_val > ema_50_1w_val or high_val >= r3_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike_v2"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-26 03:33

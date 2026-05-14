# Strategy: 6h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.594 | +49.7% | -8.2% | 200 | PASS |
| ETHUSDT | 0.085 | +23.7% | -10.1% | 176 | PASS |
| SOLUSDT | 0.818 | +109.1% | -23.9% | 149 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.301 | -6.8% | -11.3% | 70 | FAIL |
| ETHUSDT | 1.456 | +31.7% | -6.9% | 61 | PASS |
| SOLUSDT | 0.382 | +11.6% | -8.9% | 50 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
6h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: On 6h timeframe, trade Camarilla R1/S1 breakouts with 1d EMA34 trend filter and volume spike confirmation. Camarilla levels provide institutional support/resistance, EMA34 filters trend direction, and volume spike confirms institutional participation. Designed for 50-150 total trades over 4 years (12-37/year) with discrete sizing to minimize fee drag. Works in bull/bear markets via trend filter and avoids low-volume false breakouts.
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
    
    # Get 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(prev_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Align HTF indicators to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of EMA(34) 1d, volume MA (20)
    start_idx = max(34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
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
        
        ema_34_1d_val = ema_34_1d_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol_spike = volume_spike[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        r4_val = r4_aligned[i]
        s4_val = s4_aligned[i]
        
        # Trend filter: price > EMA34 (uptrend) or < EMA34 (downtrend)
        uptrend = close_val > ema_34_1d_val
        downtrend = close_val < ema_34_1d_val
        
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
            if close_val < ema_34_1d_val or low_val <= s3_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend reversal or price reaches R3 (mean reversion target)
            if close_val > ema_34_1d_val or high_val >= r3_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-26 03:32

# Strategy: 12h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.085 | +17.9% | -8.9% | 71 | FAIL |
| ETHUSDT | 0.038 | +21.8% | -7.3% | 59 | PASS |
| SOLUSDT | 0.325 | +43.0% | -23.9% | 61 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.137 | +7.4% | -5.6% | 24 | PASS |
| SOLUSDT | -1.256 | -10.1% | -17.3% | 23 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
12h Camarilla H3/L3 Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Camarilla H3 (resistance) and L3 (support) levels from daily pivots act as 
institutional order flow zones. Breaking above H3 with volume and daily uptrend signals 
bullish momentum; breaking below L3 with volume and daily downtrend signals bearish 
momentum. Works in bull/bear markets by only taking breakouts aligned with daily EMA34 
trend. Uses 12h timeframe to reduce trade frequency (target: 12-37/year) and minimize 
fee drag while capturing multi-day moves.
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
    
    # Get 1d data for EMA34 trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla pivot levels
    # Typical price = (high + low + close) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Camarilla pivot = typical price
    pivot = typical_price.values
    # Ranges
    hl_range = df_1d['high'].values - df_1d['low'].values
    # H3 = pivot + (high - low) * 1.1 / 4
    # L3 = pivot - (high - low) * 1.1 / 4
    camarilla_h3 = pivot + hl_range * 1.1 / 4
    camarilla_l3 = pivot - hl_range * 1.1 / 4
    
    # Align HTF levels to LTF with proper delay
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA34 warmup
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_34_aligned[i]
        h3_level = camarilla_h3_aligned[i]
        l3_level = camarilla_l3_aligned[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Breakout signals with trend filter
        if position == 0:
            # Long: price breaks above H3 (resistance) AND above daily EMA34 (uptrend filter)
            long_condition = (curr_close > h3_level) and (curr_close > ema_trend) and volume_spike
            # Short: price breaks below L3 (support) AND below daily EMA34 (downtrend filter)
            short_condition = (curr_close < l3_level) and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to L3 or trend breaks
            if curr_close <= l3_level or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to H3 or trend breaks
            if curr_close >= h3_level or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-25 01:17

# Strategy: 4h_Camarilla_R1S1_Breakout_Volume_EMATrend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.401 | +39.4% | -8.8% | 210 | PASS |
| ETHUSDT | 0.000 | +18.8% | -11.7% | 205 | PASS |
| SOLUSDT | 0.978 | +139.2% | -21.6% | 171 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.606 | -9.1% | -9.3% | 85 | FAIL |
| ETHUSDT | 0.978 | +21.9% | -8.2% | 71 | PASS |
| SOLUSDT | -0.072 | +4.2% | -8.5% | 57 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h Camarilla Pivot Breakout with Volume Spike and Daily EMA Trend
Hypothesis: Camarilla pivot levels (R1, S1) from daily chart act as key support/resistance.
Breakouts beyond these levels with volume confirmation and trend alignment capture momentum.
Works in bull/bear markets by requiring volume spike and daily EMA trend filter to avoid false breakouts.
Designed for 20-50 trades/year on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation (once before loop)
    df_d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for previous day
    # Formula: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C = (H+L+C)/3 (typical price)
    typical_price = (df_d['high'] + df_d['low'] + df_d['close']) / 3
    range_hl = df_d['high'] - df_d['low']
    r1 = typical_price + (range_hl * 1.1 / 12)
    s1 = typical_price - (range_hl * 1.1 / 12)
    
    # Shift by 1 to avoid look-ahead (use previous day's levels)
    r1_prev = r1.shift(1).values
    s1_prev = s1.shift(1).values
    
    # Align to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_d, r1_prev)
    s1_aligned = align_htf_to_ltf(prices, df_d, s1_prev)
    
    # Daily EMA34 for trend filter
    ema_34_d = pd.Series(df_d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_d, ema_34_d)
    
    # Volume spike: 2x 20-period average on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_trend = ema_34_aligned[i]
        
        if position == 0:
            # Long: break above R1 with volume spike and price above daily EMA (uptrend)
            if price > r1_val and volume_spike[i] and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume spike and price below daily EMA (downtrend)
            elif price < s1_val and volume_spike[i] and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: price returns to S1 or breaks below daily EMA
            if price <= s1_val or price < ema_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: price returns to R1 or breaks above daily EMA
            if price >= r1_val or price > ema_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_Volume_EMATrend"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-18 01:31

# Strategy: 6h_chaikin_money_flow_1d_trend_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.266 | +31.6% | -6.9% | 24 | PASS |
| ETHUSDT | 0.246 | +33.5% | -11.3% | 30 | PASS |
| SOLUSDT | 1.536 | +310.7% | -20.2% | 25 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.541 | +1.4% | -6.2% | 8 | FAIL |
| ETHUSDT | 0.221 | +8.4% | -7.7% | 6 | PASS |
| SOLUSDT | -1.168 | -10.4% | -12.3% | 9 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h_chaikin_money_flow_1d_trend_volume_v1
Hypothesis: Chaikin Money Flow (CMF) detects institutional buying/selling pressure. 
Long when CMF > 0.15 and price above 1d EMA50 (accumulation + uptrend).
Short when CMF < -0.15 and price below 1d EMA50 (distribution + downtrend).
Uses 6h for timing, 1d for trend and CMF filter. Works in bull/bear by following institutional flow.
Target: 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_chaikin_money_flow_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for CMF and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate CMF(20) on 1d
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # Avoid division by zero
    hl_range = df_1d['high'] - df_1d['low']
    hl_range = hl_range.replace(0, np.nan)
    mfm = ((df_1d['close'] - df_1d['low']) - (df_1d['high'] - df_1d['close'])) / hl_range
    mfm = mfm.fillna(0)  # when hl_range=0, set mfm=0
    
    # Money Flow Volume = MFM * Volume
    mfv = mfm * df_1d['volume']
    
    # CMF = 20-period sum of MFV / 20-period sum of Volume
    mfv_sum = mfv.rolling(window=20, min_periods=20).sum()
    vol_sum = df_1d['volume'].rolling(window=20, min_periods=20).sum()
    cmf = mfv_sum / vol_sum
    cmf = cmf.replace([np.inf, -np.inf], 0).fillna(0)
    
    # 1d EMA50 for trend filter
    ema_50 = df_1d['close'].ewm(span=50, adjust=False).mean()
    
    # Align all 1d data to 6h timeframe
    cmf_aligned = align_htf_to_ltf(prices, df_1d, cmf.values)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50.values)
    
    # Volume confirmation (20-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(cmf_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: CMF turns negative or price breaks below EMA50
            if cmf_aligned[i] < 0 or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: CMF turns positive or price breaks above EMA50
            if cmf_aligned[i] > 0 or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: CMF > 0.15 with volume and price above EMA50
            if (cmf_aligned[i] > 0.15 and vol_confirm and 
                close[i] > ema_50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: CMF < -0.15 with volume and price below EMA50
            elif (cmf_aligned[i] < -0.15 and vol_confirm and 
                  close[i] < ema_50_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-07 15:21

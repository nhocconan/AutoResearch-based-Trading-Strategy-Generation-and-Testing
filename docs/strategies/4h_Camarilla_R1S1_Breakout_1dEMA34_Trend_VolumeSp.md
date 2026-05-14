# Strategy: 4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSp

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.511 | +38.7% | -5.9% | 261 | PASS |
| ETHUSDT | 0.173 | +27.7% | -12.1% | 250 | PASS |
| SOLUSDT | 0.556 | +60.1% | -13.3% | 221 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.856 | -7.3% | -9.3% | 105 | FAIL |
| ETHUSDT | 0.236 | +8.4% | -10.1% | 89 | PASS |
| SOLUSDT | 0.772 | +14.0% | -5.1% | 73 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSp
Hypothesis: 4h Camarilla R1/S1 breakout with 1-day EMA34 trend filter and volume confirmation.
Long when price breaks above R1 with 1d uptrend and volume spike. Short when price breaks below S1 with 1d downtrend and volume spike.
Camarilla levels provide precise intraday support/resistance. EMA34 filter ensures trading with higher timeframe trend.
Volume confirmation reduces false breakouts. Target: 20-50 trades/year (80-200 over 4 years).
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
    
    # Daily data for Camarilla calculation (based on prior day)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for current day using prior day's OHLC
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), 
    #            R2 = close + 0.55*(high-low), R1 = close + 0.275*(high-low)
    #            S1 = close - 0.275*(high-low), S2 = close - 0.55*(high-low)
    #            S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    range_1d = high_1d - low_1d
    camarilla_r1 = close_1d + 0.275 * range_1d
    camarilla_s1 = close_1d - 0.275 * range_1d
    
    # Align Camarilla levels to 4h timeframe (shifted by 1 day for proper timing)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34(1d) and volume MA
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 + 1d uptrend + volume spike
            long_setup = (close[i] > camarilla_r1_aligned[i]) and (close[i] > ema_34_1d_aligned[i]) and volume_spike[i]
            # Short: break below S1 + 1d downtrend + volume spike
            short_setup = (close[i] < camarilla_s1_aligned[i]) and (close[i] < ema_34_1d_aligned[i]) and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.30
                position = 1
            elif short_setup:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.30
            # Exit: price closes below R1 OR 1d trend turns down
            if (close[i] < camarilla_r1_aligned[i]) or (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            # Exit: price closes above S1 OR 1d trend turns up
            if (close[i] > camarilla_s1_aligned[i]) or (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSp"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 12:36

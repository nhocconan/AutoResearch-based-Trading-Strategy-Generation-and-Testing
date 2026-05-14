# Strategy: 6h_Camarilla_R3S3_12hEMA34_Trend_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.292 | +5.4% | -9.9% | 95 | FAIL |
| ETHUSDT | 0.105 | +24.7% | -19.4% | 78 | PASS |
| SOLUSDT | 0.115 | +23.6% | -25.8% | 69 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.045 | +6.0% | -9.0% | 28 | PASS |
| SOLUSDT | -0.561 | -4.2% | -14.1% | 22 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 12h EMA34 for trend filter (price > EMA34 = uptrend, price < EMA34 = downtrend).
- Entry: Long when price breaks above Camarilla R3 AND price > 12h EMA34 AND volume > 2.0 * 6h volume MA(20);
         Short when price breaks below Camarilla S3 AND price < 12h EMA34 AND volume > 2.0 * 6h volume MA(20).
- Exit: Opposite Camarilla breakout (Long exits when price < Camarilla S1, Short exits when price > Camarilla R1).
- Signal size: 0.25 discrete to balance capture and fee control.
- Camarilla levels provide intraday structure, EMA34 filters higher-timeframe trend, volume spike confirms conviction.
- Works in bull (buying breakouts) and bear (selling breakdowns) with reduced whipsaws from 12h trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34
    close_12h = df_12h['close'].values
    ema_34 = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34)
    
    # Get 1d data for Camarilla pivot levels (based on previous day OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d OHLC
    # Camarilla: H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    #            H3 = close + 1.25*(high-low), L3 = close - 1.25*(high-low)
    #            H2 = close + 1.166*(high-low), L2 = close - 1.166*(high-low)
    #            H1 = close + 1.0833*(high-low), L1 = close - 1.0833*(high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ranges
    range_1d = high_1d - low_1d
    
    # Calculate Camarilla levels
    camarilla_h4 = close_1d + 1.5 * range_1d
    camarilla_l4 = close_1d - 1.5 * range_1d
    camarilla_h3 = close_1d + 1.25 * range_1d
    camarilla_l3 = close_1d - 1.25 * range_1d
    camarilla_h2 = close_1d + 1.166 * range_1d
    camarilla_l2 = close_1d - 1.166 * range_1d
    camarilla_h1 = close_1d + 1.0833 * range_1d
    camarilla_l1 = close_1d - 1.0833 * range_1d
    
    # Align Camarilla levels to 6h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    
    # Get 6h data for volume MA(20)
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 1)  # EMA34 needs 34 periods
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_h1_aligned[i]) or 
            np.isnan(camarilla_l1_aligned[i]) or np.isnan(vol_ma_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter: price > EMA34 = uptrend, price < EMA34 = downtrend
        uptrend = curr_close > ema_34_aligned[i]
        downtrend = curr_close < ema_34_aligned[i]
        
        # Volume confirmation: 2.0x threshold
        vol_confirm = curr_volume > 2.0 * vol_ma_6h[i]
        
        if position == 0:
            # Check for entry signals
            if uptrend and vol_confirm:
                # Long: price breaks above Camarilla R3 (H3)
                if curr_high > camarilla_h3_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            elif downtrend and vol_confirm:
                # Short: price breaks below Camarilla S3 (L3)
                if curr_low < camarilla_l3_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when price breaks below Camarilla S1 (L1)
            if curr_low < camarilla_l1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price breaks above Camarilla R1 (H1)
            if curr_high > camarilla_h1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_12hEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-24 17:11

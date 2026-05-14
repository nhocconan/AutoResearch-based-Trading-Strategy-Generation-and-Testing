# Strategy: 4h_Camarilla_R1S1_1dEMA34_Trend_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.205 | +31.6% | -16.3% | 83 | PASS |
| ETHUSDT | 0.117 | +24.8% | -17.3% | 85 | PASS |
| SOLUSDT | 0.866 | +168.2% | -34.8% | 82 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.419 | -0.0% | -7.4% | 35 | FAIL |
| ETHUSDT | 0.766 | +22.4% | -9.4% | 26 | PASS |
| SOLUSDT | 0.310 | +11.7% | -13.3% | 24 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d EMA34 for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Camarilla pivot levels: Calculated from prior 1d OHLC (R1, S1, R3, S3, H3, L3).
- Entry: Long when price breaks above prior 1d R1 AND 1d EMA34 bullish AND volume > 2.0 * volume MA(20).
         Short when price breaks below prior 1d S1 AND 1d EMA34 bearish AND volume > 2.0 * volume MA(20).
- Exit: Close-based reversal - exit long when price crosses below 1d EMA34,
        exit short when price crosses above 1d EMA34.
- Signal size: 0.25 discrete to balance return and drawdown.
This strategy captures medium-term breakouts aligned with the 1d trend, designed to work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate prior 1d Camarilla levels (R1, S1, R3, S3, H3, L3)
    # Using prior 1d candle to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations
    rang = high_1d - low_1d
    camarilla_h3 = close_1d + rang * 1.1 / 4
    camarilla_l3 = close_1d - rang * 1.1 / 4
    camarilla_h4 = close_1d + rang * 1.1 / 2
    camarilla_l4 = close_1d - rang * 1.1 / 2
    camarilla_r3 = close_1d + rang * 1.1 / 6
    camarilla_s3 = close_1d - rang * 1.1 / 6
    camarilla_r1 = close_1d + rang * 1.1 / 12
    camarilla_s1 = close_1d - rang * 1.1 / 12
    
    # Align HTF indicators to 4h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate volume MA(20) for confirmation (using 4h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 40)  # Need enough bars for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (2.0x threshold)
            vol_confirmed = curr_volume > 2.0 * vol_ma[i]
            
            # Long: Price breaks above prior 1d R1 AND 1d EMA34 bullish AND volume confirmed
            if curr_close > camarilla_r1_aligned[i] and curr_close > ema_1d_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below prior 1d S1 AND 1d EMA34 bearish AND volume confirmed
            elif curr_close < camarilla_s1_aligned[i] and curr_close < ema_1d_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below 1d EMA34 (trend change)
            if curr_close < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above 1d EMA34 (trend change)
            if curr_close > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-24 14:40

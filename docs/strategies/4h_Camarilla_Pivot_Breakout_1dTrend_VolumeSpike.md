# Strategy: 4h_Camarilla_Pivot_Breakout_1dTrend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.133 | +11.5% | -16.6% | 98 | FAIL |
| ETHUSDT | 0.134 | +26.5% | -20.1% | 103 | PASS |
| SOLUSDT | 0.721 | +111.7% | -26.6% | 81 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.009 | +27.4% | -8.6% | 29 | PASS |
| SOLUSDT | 0.850 | +25.2% | -9.6% | 23 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_Breakout_1dTrend_VolumeSpike
Hypothesis: Uses 1-day Camarilla pivot levels (R1/S1 for mean reversion, R4/S4 for breakout) on the 4-hour timeframe.
Trades in the direction of the 1-day EMA50 trend with volume spike confirmation to filter false breakouts.
Designed to work in both bull and bear markets by adapting to price action relative to daily pivots and trend.
Targets 20-50 trades per year to minimize fee dust.
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
    
    # Get 1-day data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous 1-day bar
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_1d = df_1d['high'] - df_1d['low']
    
    # Camarilla levels
    R4 = typical_price + (range_1d * 1.1 / 2)
    R3 = typical_price + (range_1d * 1.1 / 4)
    R1 = typical_price + (range_1d * 1.1 / 12)
    S1 = typical_price - (range_1d * 1.1 / 12)
    S3 = typical_price - (range_1d * 1.1 / 4)
    S4 = typical_price - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4.values)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3.values)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1.values)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1.values)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3.values)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4.values)
    
    # Calculate volume spike (>1.8x 20-period MA for stricter filtering)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend direction from 1-day EMA50
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = vol_spike[i]
        
        # Price relative to Camarilla levels
        price_above_R1 = close[i] > R1_aligned[i]
        price_below_S1 = close[i] < S1_aligned[i]
        price_above_R3 = close[i] > R3_aligned[i]
        price_below_S3 = close[i] < S3_aligned[i]
        price_above_R4 = close[i] > R4_aligned[i]
        price_below_S4 = close[i] < S4_aligned[i]
        
        # Entry logic:
        # Long: Mean reversion from S1/S3 OR breakout above R4 in uptrend
        long_entry = vol_confirm and trend_up and (
            (price_below_S1 and close[i] > S1_aligned[i-1]) or  # Rejection of S1
            (price_below_S3 and close[i] > S3_aligned[i-1]) or  # Rejection of S3
            (price_above_R4 and close[i-1] <= R4_aligned[i-1])   # Breakout above R4
        )
        
        # Short: Mean reversion from R1/R3 OR breakdown below S4 in downtrend
        short_entry = vol_confirm and trend_down and (
            (price_above_R1 and close[i] < R1_aligned[i-1]) or  # Rejection of R1
            (price_above_R3 and close[i] < R3_aligned[i-1]) or  # Rejection of R3
            (price_below_S4 and close[i-1] >= S4_aligned[i-1])   # Breakdown below S4
        )
        
        # Exit logic: Opposite level rejection or trend reversal
        long_exit = (price_above_R3 and close[i] < R3_aligned[i-1]) or not trend_up
        short_exit = (price_below_S1 and close[i] > S1_aligned[i-1]) or not trend_down
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_Pivot_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-28 02:31

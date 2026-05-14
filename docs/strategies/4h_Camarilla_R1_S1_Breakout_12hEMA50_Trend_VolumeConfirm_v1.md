# Strategy: 4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.467 | +43.0% | -12.4% | 351 | PASS |
| ETHUSDT | 0.136 | +26.6% | -12.8% | 321 | PASS |
| SOLUSDT | 0.447 | +57.1% | -20.3% | 258 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.910 | -11.5% | -11.7% | 132 | FAIL |
| ETHUSDT | 1.023 | +22.3% | -10.1% | 114 | PASS |
| SOLUSDT | -0.003 | +5.3% | -9.7% | 97 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume spike confirmation.
# Uses Camarilla R1/S1 levels from 12h pivots for precise breakout entries, 12h EMA50 for trend alignment,
# and volume spike (>2.0x 20-bar MA) for confirmation. Designed for 4h timeframe to achieve 75-200
# total trades over 4 years (19-50/year) with discrete sizing (0.30) to minimize fee drag.
# Works in both bull and bear markets via trend filter and tight entry conditions.

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h Camarilla pivot levels (R1, S1) for breakout
    # Based on previous 12h bar's high, low, close
    prev_12h_high = df_12h['high'].shift(1).values
    prev_12h_low = df_12h['low'].shift(1).values
    prev_12h_close = df_12h['close'].shift(1).values
    
    camarilla_r1 = prev_12h_close + (prev_12h_high - prev_12h_low) * 1.1 / 12
    camarilla_s1 = prev_12h_close - (prev_12h_high - prev_12h_low) * 1.1 / 12
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume on 4h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, 20) + 1  # 51 (for EMA50 and volume MA)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter: 12h EMA50 direction
        uptrend = curr_close > ema_50_12h_aligned[i]
        downtrend = curr_close < ema_50_12h_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 4h Camarilla R1/S1 breakout conditions
        breakout_r1 = curr_high > camarilla_r1_aligned[i]  # Break above 12h R1
        breakdown_s1 = curr_low < camarilla_s1_aligned[i]  # Break below 12h S1
        
        if position == 0:  # Flat - look for new entries
            # Long: 12h R1 breakout AND uptrend AND volume confirmation
            if breakout_r1 and uptrend and vol_confirm:
                signals[i] = 0.30
                position = 1
            # Short: 12h S1 breakdown AND downtrend AND volume confirmation
            elif breakdown_s1 and downtrend and vol_confirm:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on 12h S1 breakdown (reversal signal)
            if curr_low < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit on 12h R1 breakout (reversal signal)
            if curr_high > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
```

## Last Updated
2026-05-01 16:23

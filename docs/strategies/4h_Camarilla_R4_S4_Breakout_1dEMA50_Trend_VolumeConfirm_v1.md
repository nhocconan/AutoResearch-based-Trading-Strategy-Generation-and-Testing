# Strategy: 4h_Camarilla_R4_S4_Breakout_1dEMA50_Trend_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.485 | +50.8% | -10.6% | 199 | PASS |
| ETHUSDT | 0.131 | +26.2% | -17.6% | 202 | PASS |
| SOLUSDT | 1.165 | +244.8% | -16.2% | 164 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.714 | -3.0% | -8.0% | 71 | FAIL |
| ETHUSDT | 0.477 | +14.8% | -12.8% | 67 | PASS |
| SOLUSDT | 0.561 | +16.9% | -10.6% | 56 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 1d EMA50 trend filter and volume confirmation.
# Uses Camarilla R4/S4 levels from 1d pivots for breakout entries (wider bands = fewer false signals),
# 1d EMA50 for trend alignment, and volume spike (>1.8x 24-bar MA) for confirmation.
# Designed for 4h timeframe to achieve 75-200 total trades over 4 years (19-50/year) with discrete sizing (0.30).
# Works in both bull and bear markets via trend filter and tight entry conditions.

name = "4h_Camarilla_R4_S4_Breakout_1dEMA50_Trend_VolumeConfirm_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d Camarilla pivot levels (R4, S4) for breakout
    # Based on previous 1d bar's high, low, close
    prev_1d_high = df_1d['high'].shift(1).values
    prev_1d_low = df_1d['low'].shift(1).values
    prev_1d_close = df_1d['close'].shift(1).values
    
    camarilla_r4 = prev_1d_close + (prev_1d_high - prev_1d_low) * 1.1 / 2
    camarilla_s4 = prev_1d_close - (prev_1d_high - prev_1d_low) * 1.1 / 2
    
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: current volume > 1.8 * 24-period average volume on 4h
    volume_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (volume_ma_24 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, 24) + 1  # 51 (for EMA50 and volume MA)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter: 1d EMA50 direction
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 4h Camarilla R4/S4 breakout conditions
        breakout_r4 = curr_high > camarilla_r4_aligned[i]  # Break above 1d R4
        breakdown_s4 = curr_low < camarilla_s4_aligned[i]  # Break below 1d S4
        
        if position == 0:  # Flat - look for new entries
            # Long: 1d R4 breakout AND uptrend AND volume confirmation
            if breakout_r4 and uptrend and vol_confirm:
                signals[i] = 0.30
                position = 1
            # Short: 1d S4 breakdown AND downtrend AND volume confirmation
            elif breakdown_s4 and downtrend and vol_confirm:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on 1d S4 breakdown (reversal signal)
            if curr_low < camarilla_s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit on 1d R4 breakout (reversal signal)
            if curr_high > camarilla_r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
```

## Last Updated
2026-05-01 16:25

# Strategy: 6h_Camarilla_R4_S4_Breakout_1dEMA50_Trend_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.398 | +40.2% | -7.8% | 110 | PASS |
| ETHUSDT | 0.177 | +29.1% | -16.1% | 98 | PASS |
| SOLUSDT | 0.645 | +90.4% | -25.4% | 87 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.488 | +0.7% | -6.3% | 37 | FAIL |
| ETHUSDT | 1.172 | +26.6% | -6.9% | 33 | PASS |
| SOLUSDT | 0.183 | +8.3% | -12.1% | 30 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R4/S4 breakout with 1d trend filter and volume confirmation.
# Uses 1d Camarilla R4/S4 levels (stronger breakout levels) to avoid false breakouts.
# Trades only in direction of 1d EMA50 trend with volume spike confirmation.
# R4/S4 breakouts indicate stronger momentum and are less prone to whipsaw.
# Works in bull (buy R4 breakout with uptrend) and bear (sell S4 breakdown with downtrend).
# Discrete position sizing 0.25 balances return and drawdown. Target: 50-150 trades over 4 years.

name = "6h_Camarilla_R4_S4_Breakout_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
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
    
    # Calculate 1d Camarilla pivot levels (R4, S4) - stronger breakout levels
    # Camarilla: based on previous day's high, low, close
    # R4 = close + (high - low) * 1.1/2
    # S4 = close - (high - low) * 1.1/2
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    camarilla_r4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume on 6h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, 20) + 1  # 51 (for EMA50 and volume MA20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 1d EMA50 direction
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Camarilla R4/S4 breakout conditions
        breakout_r4 = curr_close > camarilla_r4_aligned[i]  # Break above R4
        breakdown_s4 = curr_close < camarilla_s4_aligned[i]  # Break below S4
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above R4 AND uptrend AND volume confirmation
            if breakout_r4 and uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below S4 AND downtrend AND volume confirmation
            elif breakdown_s4 and downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on breakdown below S4 (reversal signal)
            if curr_close < camarilla_s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on breakout above R4 (reversal signal)
            if curr_close > camarilla_r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-01 15:48

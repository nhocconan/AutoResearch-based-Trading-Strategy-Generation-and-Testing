# Strategy: 4h_Camarilla_R4_S4_Breakout_12hEMA50_Trend_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.427 | +33.6% | -4.6% | 241 | PASS |
| ETHUSDT | 0.459 | +37.8% | -7.2% | 231 | PASS |
| SOLUSDT | 0.546 | +53.4% | -16.9% | 191 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.227 | -0.7% | -4.7% | 89 | FAIL |
| ETHUSDT | 0.567 | +11.7% | -5.7% | 79 | PASS |
| SOLUSDT | -0.224 | +4.0% | -6.5% | 70 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 12h EMA50 trend filter and volume confirmation.
# Uses tighter Camarilla levels (R4/S4) for stronger breakouts, 12h EMA50 for trend, and volume spike for confirmation.
# Designed for lower trade frequency (<150/year) to avoid fee drift, works in both bull and bear via trend alignment.
# Discrete position sizing 0.25 to balance return and drawdown.

name = "4h_Camarilla_R4_S4_Breakout_12hEMA50_Trend_VolumeConfirm_v1"
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
    
    # Calculate 1d Camarilla pivot levels (R4, S4) - stronger breakout levels
    # Camarilla: based on previous day's high, low, close
    # R4 = close + (high - low) * 1.1/2
    # S4 = close - (high - low) * 1.1/2
    prev_daily_high = df_12h['high'].shift(1).values
    prev_daily_low = df_12h['low'].shift(1).values
    prev_daily_close = df_12h['close'].shift(1).values
    
    camarilla_r4 = prev_daily_close + (prev_daily_high - prev_daily_low) * 1.1 / 2
    camarilla_s4 = prev_daily_close - (prev_daily_high - prev_daily_low) * 1.1 / 2
    
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # 4h Camarilla pivot levels (R4, S4) for breakout
    # Based on previous 4h bar's high, low, close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_r4_4h = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s4_4h = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume on 4h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, 20) + 1  # 51 (for EMA50 and volume MA)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(camarilla_r4_4h[i]) or
            np.isnan(camarilla_s4_4h[i]) or
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
        
        # 4h Camarilla R4/S4 breakout conditions
        breakout_r4 = curr_high > camarilla_r4_4h[i]  # Break above 4h R4
        breakdown_s4 = curr_low < camarilla_s4_4h[i]  # Break below 4h S4
        
        # Daily Camarilla R4/S4 confirmation
        confirm_r4 = curr_close > camarilla_r4_aligned[i]  # Confirm above daily R4
        confirm_s4 = curr_close < camarilla_s4_aligned[i]  # Confirm below daily S4
        
        if position == 0:  # Flat - look for new entries
            # Long: 4h R4 breakout AND daily R4 confirmation AND uptrend AND volume confirmation
            if breakout_r4 and confirm_r4 and uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: 4h S4 breakdown AND daily S4 confirmation AND downtrend AND volume confirmation
            elif breakdown_s4 and confirm_s4 and downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on 4h S4 breakdown (reversal signal)
            if curr_low < camarilla_s4_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on 4h R4 breakout (reversal signal)
            if curr_high > camarilla_r4_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-01 16:19

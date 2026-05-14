# Strategy: 4h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeConfirm_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.401 | +34.0% | -9.7% | 533 | PASS |
| ETHUSDT | 0.002 | +20.8% | -7.9% | 514 | PASS |
| SOLUSDT | 0.088 | +23.6% | -22.2% | 521 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.626 | -4.1% | -7.1% | 196 | FAIL |
| ETHUSDT | 0.088 | +6.8% | -8.1% | 182 | PASS |
| SOLUSDT | 0.257 | +8.5% | -4.7% | 180 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d trend filter and volume confirmation.
# Uses 1d EMA34 as trend filter and 1d ATR for volatility-based stoploss.
# Works in bull (buy R3 breakout with uptrend) and bear (sell S3 breakdown with downtrend).
# Discrete position sizing 0.25 balances return and drawdown. Target: 75-200 trades over 4 years.

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeConfirm_v2"
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
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla pivot levels (R3, S3) - strong breakout levels
    # Camarilla: based on previous day's high, low, close
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    prev_daily_high = df_1d['high'].shift(1).values
    prev_daily_low = df_1d['low'].shift(1).values
    prev_daily_close = df_1d['close'].shift(1).values
    
    camarilla_r3 = prev_daily_close + (prev_daily_high - prev_daily_low) * 1.1 / 4
    camarilla_s3 = prev_daily_close - (prev_daily_high - prev_daily_low) * 1.1 / 4
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 4h Camarilla pivot levels (R3, S3) for breakout
    # Based on previous 4h bar's high, low, close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_r3_4h = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3_4h = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume on 4h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(34, 20) + 1  # 35 (for EMA34 and volume MA)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_r3_4h[i]) or
            np.isnan(camarilla_s3_4h[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter: 1d EMA34 direction
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 4h Camarilla breakout conditions
        breakout_r3 = curr_high > camarilla_r3_4h[i]  # Break above 4h R3
        breakdown_s3 = curr_low < camarilla_s3_4h[i]  # Break below 4h S3
        
        # Daily Camarilla R3/S3 confirmation
        confirm_r3 = curr_close > camarilla_r3_aligned[i]  # Confirm above daily R3
        confirm_s3 = curr_close < camarilla_s3_aligned[i]  # Confirm below daily S3
        
        if position == 0:  # Flat - look for new entries
            # Long: 4h R3 breakout AND daily R3 confirmation AND uptrend AND volume confirmation
            if breakout_r3 and confirm_r3 and uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: 4h S3 breakdown AND daily S3 confirmation AND downtrend AND volume confirmation
            elif breakdown_s3 and confirm_s3 and downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on 4h S3 breakdown (reversal signal)
            if curr_low < camarilla_s3_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on 4h R3 breakout (reversal signal)
            if curr_high > camarilla_r3_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-01 16:16

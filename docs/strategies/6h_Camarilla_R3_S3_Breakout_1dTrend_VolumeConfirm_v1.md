# Strategy: 6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.383 | +42.1% | -11.5% | 137 | PASS |
| ETHUSDT | 0.205 | +31.8% | -13.1% | 147 | PASS |
| SOLUSDT | 1.022 | +183.3% | -29.8% | 135 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.061 | -6.2% | -10.6% | 55 | FAIL |
| ETHUSDT | 0.097 | +6.7% | -10.6% | 48 | PASS |
| SOLUSDT | 0.216 | +9.1% | -9.6% | 46 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d trend filter and volume confirmation.
# Uses Camarilla pivot levels from 1d timeframe to identify key support/resistance.
# Breakouts above R3 or below S3 are traded in the direction of 1d EMA34 trend.
# Volume confirmation ensures breakouts have sufficient participation.
# Works in both bull (buy R3 breakout with uptrend) and bear (sell S3 breakdown with downtrend).
# Discrete position sizing (0.25) balances return and drawdown. Target: 50-150 trades over 4 years.

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeConfirm_v1"
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
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d ATR(14) for volatility filter (optional, can use volume instead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 1d Camarilla pivot levels
    # Camarilla: based on previous day's high, low, close
    # R4 = close + (high - low) * 1.5/2
    # R3 = close + (high - low) * 1.25/2
    # S3 = close - (high - low) * 1.25/2
    # S4 = close - (high - low) * 1.5/2
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.25 / 2
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.25 / 2
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume on 6h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(35, 20) + 1  # 36 (for EMA34 and volume MA20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 1d EMA34 direction
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Camarilla breakout conditions
        breakout_r3 = curr_close > camarilla_r3_aligned[i]  # Break above R3
        breakdown_s3 = curr_close < camarilla_s3_aligned[i]  # Break below S3
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above R3 AND uptrend AND volume confirmation
            if breakout_r3 and uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below S3 AND downtrend AND volume confirmation
            elif breakdown_s3 and downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on breakdown below S3 (reversal signal)
            if curr_close < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on breakout above R3 (reversal signal)
            if curr_close > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-01 15:27

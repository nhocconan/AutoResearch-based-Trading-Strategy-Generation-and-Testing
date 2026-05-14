# Strategy: 4h_Camarilla_R1_S1_Breakout_1dEMA50_Trend_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.115 | +25.4% | -9.1% | 299 | PASS |
| ETHUSDT | -0.640 | -13.6% | -21.7% | 288 | FAIL |
| SOLUSDT | 1.072 | +169.0% | -18.0% | 244 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.074 | +6.6% | -5.6% | 93 | PASS |
| SOLUSDT | 0.062 | +6.2% | -10.5% | 92 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA50 trend filter and volume confirmation.
# Uses Camarilla pivot levels from 1d timeframe to identify key intraday support/resistance.
# Breakouts above R1 or below S1 are traded in the direction of 1d EMA50 trend.
# Volume confirmation ensures breakouts have sufficient participation.
# Works in both bull (buy R1 breakout with uptrend) and bear (sell S1 breakdown with downtrend).
# Discrete position sizing (0.25) balances return and drawdown. Target: 75-200 trades over 4 years.

name = "4h_Camarilla_R1_S1_Breakout_1dEMA50_Trend_VolumeConfirm_v1"
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
    if len(df_1d) < 55:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d ATR(14) for volatility (optional, using volume instead)
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
    
    # Calculate 1d Camarilla pivot levels (R1, S1)
    # Camarilla: based on previous day's high, low, close
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume on 4h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(55, 20) + 1  # 56 (for EMA50 and volume MA20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
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
        
        # Camarilla breakout conditions
        breakout_r1 = curr_close > camarilla_r1_aligned[i]  # Break above R1
        breakdown_s1 = curr_close < camarilla_s1_aligned[i]  # Break below S1
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above R1 AND uptrend AND volume confirmation
            if breakout_r1 and uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below S1 AND downtrend AND volume confirmation
            elif breakdown_s1 and downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on breakdown below S1 (reversal signal)
            if curr_close < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on breakout above R1 (reversal signal)
            if curr_close > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-01 15:41

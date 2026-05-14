# Strategy: 4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.295 | +32.8% | -7.8% | 317 | PASS |
| ETHUSDT | 0.107 | +24.9% | -11.1% | 282 | PASS |
| SOLUSDT | 0.465 | +55.5% | -20.0% | 236 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.669 | -7.5% | -8.0% | 115 | FAIL |
| ETHUSDT | 1.283 | +24.5% | -6.1% | 94 | PASS |
| SOLUSDT | 0.104 | +7.0% | -8.1% | 83 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation (>2x 20-bar MA)
# Uses 12h HTF for stronger trend alignment than 1d, reducing whipsaws in ranging markets while maintaining sufficient signal frequency.
# Camarilla breakouts capture strong momentum moves after range-bound periods.
# Volume confirmation with higher threshold (>2x) ensures institutional participation and reduces false breakouts.
# Discrete sizing (0.25) minimizes fee churn. Target: 75-200 total trades over 4 years (19-50/year) with strong BTC/ETH performance.

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h HTF data for EMA calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA(50) on 12h close
    ema_12h_50 = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA to 4h timeframe
    ema_12h_50_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_50)
    
    # Calculate Camarilla pivot levels from previous 12h bar
    # Need high, low, close from previous 12h bar
    prev_12h_high = df_12h['high'].shift(1).values
    prev_12h_low = df_12h['low'].shift(1).values
    prev_12h_close = df_12h['close'].shift(1).values
    
    # Camarilla R3, S3 levels
    camarilla_r3 = prev_12h_close + (prev_12h_high - prev_12h_low) * 1.1 / 4
    camarilla_s3 = prev_12h_close - (prev_12h_high - prev_12h_low) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 20  # Need 20 for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_12h_50_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_ma_20[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3, above 12h EMA, and volume confirmation
            if curr_high > camarilla_r3_aligned[i] and curr_close > ema_12h_50_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3, below 12h EMA, and volume confirmation
            elif curr_low < camarilla_s3_aligned[i] and curr_close < ema_12h_50_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on price breaking below Camarilla S3 or below 12h EMA
            if curr_low < camarilla_s3_aligned[i] or curr_close < ema_12h_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on price breaking above Camarilla R3 or above 12h EMA
            if curr_high > camarilla_r3_aligned[i] or curr_close > ema_12h_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-01 17:01

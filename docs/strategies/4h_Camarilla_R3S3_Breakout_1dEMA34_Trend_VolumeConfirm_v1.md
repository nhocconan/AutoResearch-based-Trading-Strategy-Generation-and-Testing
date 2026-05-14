# Strategy: 4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.184 | +29.0% | -9.4% | 360 | PASS |
| ETHUSDT | 0.043 | +20.6% | -14.5% | 340 | PASS |
| SOLUSDT | 0.787 | +113.8% | -18.4% | 311 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.845 | -2.9% | -9.1% | 129 | FAIL |
| ETHUSDT | 0.346 | +11.2% | -9.0% | 112 | PASS |
| SOLUSDT | 0.497 | +14.2% | -7.8% | 113 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation (>1.5x 20-bar MA)
# Uses 1d HTF for stronger trend alignment than 4h, reducing whipsaws in ranging markets.
# Camarilla breakouts capture strong momentum moves after range-bound periods.
# Volume confirmation ensures institutional participation. Discrete sizing (0.25) minimizes fee churn.
# Target: 75-200 total trades over 4 years (19-50/year) with strong BTC/ETH performance.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeConfirm_v1"
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
    
    # 1d HTF data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(34) on 1d close
    ema_1d_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 4h timeframe
    ema_1d_34_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_34)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Need high, low, close from previous 1d bar
    prev_1d_high = df_1d['high'].shift(1).values
    prev_1d_low = df_1d['low'].shift(1).values
    prev_1d_close = df_1d['close'].shift(1).values
    
    # Camarilla R3, S3 levels
    camarilla_r3 = prev_1d_close + (prev_1d_high - prev_1d_low) * 1.1 / 4
    camarilla_s3 = prev_1d_close - (prev_1d_high - prev_1d_low) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 20  # Need 20 for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_1d_34_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_ma_20[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3, above 1d EMA, and volume confirmation
            if curr_high > camarilla_r3_aligned[i] and curr_close > ema_1d_34_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3, below 1d EMA, and volume confirmation
            elif curr_low < camarilla_s3_aligned[i] and curr_close < ema_1d_34_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on price breaking below Camarilla S3 or below 1d EMA
            if curr_low < camarilla_s3_aligned[i] or curr_close < ema_1d_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on price breaking above Camarilla R3 or above 1d EMA
            if curr_high > camarilla_r3_aligned[i] or curr_close > ema_1d_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-01 16:59

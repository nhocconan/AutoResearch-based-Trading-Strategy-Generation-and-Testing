# Strategy: 6h_Camarilla_R3S3_1wEMA34_Trend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.558 | +39.5% | -4.7% | 93 | PASS |
| ETHUSDT | 0.081 | +23.7% | -11.3% | 76 | PASS |
| SOLUSDT | 0.612 | +61.3% | -15.4% | 58 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.703 | -3.3% | -7.2% | 27 | FAIL |
| ETHUSDT | 0.303 | +8.8% | -6.4% | 22 | PASS |
| SOLUSDT | -1.273 | -5.3% | -12.9% | 20 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation
# Uses 6h primary timeframe targeting 12-37 trades/year (50-150 total over 4 years)
# 1w EMA34 ensures alignment with major trend to avoid counter-trend entries
# Camarilla R3/S3 levels from 1d provide significant pivot-based breakout levels
# Volume spike (>2.0 * 20-period EMA on 6h) confirms strong participation
# Discrete position sizing (0.25) minimizes fee churn while maintaining adequate exposure
# Works in bull (continuation) and bear (mean reversion via short) markets

name = "6h_Camarilla_R3S3_1wEMA34_Trend_Volume"
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
    
    # 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar (HLC of completed 1d bar)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    close_1d_shifted = np.roll(close_1d, 1)  # Previous 1d bar's close
    close_1d_shifted[0] = close_1d[0]  # First bar uses its own close
    
    # Camarilla R3, S3 levels: based on previous 1d bar's range
    # R3 = close + 1.1 * (high - low) / 4
    # S3 = close - 1.1 * (high - low) / 4
    camarilla_r3 = close_1d_shifted + 1.1 * (high_1d - low_1d) / 4
    camarilla_s3 = close_1d_shifted - 1.1 * (high_1d - low_1d) / 4
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA34 to 6h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA (6h)
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient data for all indicators
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1w EMA34
        bullish_bias = close[i] > ema_34_1w_aligned[i]
        bearish_bias = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias:
                # Long: price breaks above Camarilla R3 with volume spike
                if close[i] > camarilla_r3_aligned[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_bias:
                # Short: price breaks below Camarilla S3 with volume spike
                if close[i] < camarilla_s3_aligned[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop around 1w EMA34
        
        elif position == 1:  # Long position
            # Exit: price breaks below Camarilla S3 or price below 1w EMA34
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Camarilla R3 or price above 1w EMA34
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-02 01:08

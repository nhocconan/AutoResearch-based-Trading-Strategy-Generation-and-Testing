# Strategy: 6h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.151 | +10.7% | -16.6% | 249 | FAIL |
| ETHUSDT | 0.128 | +26.1% | -15.6% | 233 | PASS |
| SOLUSDT | 0.960 | +166.5% | -33.1% | 219 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.043 | +27.6% | -9.3% | 78 | PASS |
| SOLUSDT | 1.028 | +27.9% | -8.9% | 73 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation.
# Long when price breaks above R3 with 12h EMA50 uptrend and volume > 1.5x 20-bar average.
# Short when price breaks below S3 with 12h EMA50 downtrend and volume spike.
# Uses ATR trailing stop (2.0x) for risk management.
# Targets 50-150 total trades over 4 years (12-37/year) with discrete position sizing (0.25).
# Camarilla levels provide intraday support/resistance; breakouts with trend/volume filter avoid false signals.
# Works in bull/bear markets by aligning with higher-timeframe trend.

name = "6h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for Camarilla levels and EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h Camarilla levels (based on previous 12h bar)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h_prev = df_12h['close'].shift(1).values  # Previous 12h close
    
    # Camarilla formulas: R3/S3 = C ± (H-L)*1.1/4
    cam_r3 = close_12h_prev + (high_12h - low_12h) * 1.1 / 4
    cam_s3 = close_12h_prev - (high_12h - low_12h) * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe
    cam_r3_aligned = align_htf_to_ltf(prices, df_12h, cam_r3)
    cam_s3_aligned = align_htf_to_ltf(prices, df_12h, cam_s3)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # ATR for trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(50, 20)  # warmup for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if np.isnan(ema_50_aligned[i]) or np.isnan(cam_r3_aligned[i]) or np.isnan(cam_s3_aligned[i]):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            continue
        
        # Regime filter: price above/below 12h EMA50 determines trend direction
        is_uptrend = close[i] > ema_50_aligned[i]
        is_downtrend = close[i] < ema_50_aligned[i]
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3 + uptrend + volume confirmation
            if curr_high > cam_r3_aligned[i] and is_uptrend and curr_volume_confirm:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_close
            # Short: price breaks below S3 + downtrend + volume confirmation
            elif curr_low < cam_s3_aligned[i] and is_downtrend and curr_volume_confirm:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Update highest high since entry
            if curr_high > highest_since_entry:
                highest_since_entry = curr_high
            
            # Trailing stop: 2.0 * ATR below highest since entry
            if curr_close < highest_since_entry - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_low < lowest_since_entry:
                lowest_since_entry = curr_low
            
            # Trailing stop: 2.0 * ATR above lowest since entry
            if curr_close > lowest_since_entry + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-30 14:26

# Strategy: 6h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.584 | +40.9% | -4.7% | 206 | PASS |
| ETHUSDT | 0.350 | +34.8% | -7.4% | 188 | PASS |
| SOLUSDT | 0.585 | +64.9% | -13.0% | 151 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.249 | -2.8% | -5.4% | 82 | FAIL |
| ETHUSDT | 1.511 | +24.8% | -5.3% | 69 | PASS |
| SOLUSDT | 0.213 | +8.2% | -4.8% | 59 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Uses Camarilla pivot levels from daily timeframe for structure, breaks above R3 or below S3 for entry,
# confirmed by 1d EMA34 trend and volume spike (>2.0x 20-bar MA). Designed for 6h timeframe to achieve
# 50-150 total trades over 4 years (12-37/year) with discrete sizing (0.25). Works in both bull and bear
# markets via volatility-based breakouts and tight entry conditions requiring confluence of structure,
# trend, and volume.

name = "6h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeConfirm_v1"
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
    
    # Daily HTF data for Camarilla pivots and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla equations
    R3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    S3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    R4 = prev_close + 1.1 * (prev_high - prev_low)
    S4 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Align daily levels to 6h timeframe (wait for completed daily bar)
    R3_6h = align_htf_to_ltf(prices, df_1d, R3)
    S3_6h = align_htf_to_ltf(prices, df_1d, S3)
    R4_6h = align_htf_to_ltf(prices, df_1d, R4)
    S4_6h = align_htf_to_ltf(prices, df_1d, S4)
    
    # Daily EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup for all indicators
    start_idx = 35  # Need 34 for EMA + 1 for Camarilla shift
    
    for i in range(start_idx, n):
        if np.isnan(R3_6h[i]) or np.isnan(S3_6h[i]) or np.isnan(ema_34_6h[i]) or np.isnan(volume_ma_20[i]):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above R3, above daily EMA34, and volume confirmation
            if curr_high > R3_6h[i] and curr_close > ema_34_6h[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Price breaks below S3, below daily EMA34, and volume confirmation
            elif curr_low < S3_6h[i] and curr_close < ema_34_6h[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on close below R3 (failed breakout) or below daily EMA34
            if curr_close < R3_6h[i] or curr_close < ema_34_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on close above S3 (failed breakdown) or above daily EMA34
            if curr_close > S3_6h[i] or curr_close > ema_34_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-01 16:33

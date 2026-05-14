# Strategy: 6h_ElderRay_BullBearPower_1dEMA34_Trend_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.225 | +31.8% | -12.6% | 140 | PASS |
| ETHUSDT | 0.091 | +23.8% | -21.0% | 127 | PASS |
| SOLUSDT | 1.183 | +226.9% | -26.8% | 105 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.011 | -6.0% | -11.2% | 58 | FAIL |
| ETHUSDT | 0.260 | +9.9% | -9.3% | 48 | PASS |
| SOLUSDT | -0.125 | +2.0% | -17.4% | 43 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d EMA34 trend filter + volume confirmation.
# Elder Ray measures bull/bear power relative to EMA13. Long when Bull Power > 0 and rising,
# Short when Bear Power < 0 and falling. Confirmed by 1d EMA34 trend and volume spike (>2.0x 20-bar MA).
# Works in both bull and bear markets via trend-following with volatility-based entries.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing (0.25).

name = "6h_ElderRay_BullBearPower_1dEMA34_Trend_VolumeConfirm_v1"
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
    
    # Daily HTF data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Daily EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Elder Ray components: EMA13 and Bull/Bear Power
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power: High - EMA13
    bear_power = low - ema_13   # Bear Power: Low - EMA13
    
    # Smooth Bull/Bear Power for trend confirmation (3-period EMA)
    bull_power_smooth = pd.Series(bull_power).ewm(span=3, adjust=False, min_periods=3).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup for all indicators
    start_idx = 20  # Need 20 for volume MA, 13 for EMA13, 3 for smoothing
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_6h[i]) or np.isnan(bull_power_smooth[i]) or np.isnan(bear_power_smooth[i]) or np.isnan(volume_ma_20[i]):
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
            # Long: Bull Power > 0 and rising, above daily EMA34, and volume confirmation
            if bull_power_smooth[i] > 0 and bull_power_smooth[i] > bull_power_smooth[i-1] and curr_close > ema_34_6h[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Bear Power < 0 and falling, below daily EMA34, and volume confirmation
            elif bear_power_smooth[i] < 0 and bear_power_smooth[i] < bear_power_smooth[i-1] and curr_close < ema_34_6h[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Bull Power <= 0 (lost bullish momentum) or below daily EMA34
            if bull_power_smooth[i] <= 0 or curr_close < ema_34_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Bear Power >= 0 (lost bearish momentum) or above daily EMA34
            if bear_power_smooth[i] >= 0 or curr_close > ema_34_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-01 16:34

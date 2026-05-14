# Strategy: 4h_Camarilla_R3S3_Breakout_12hTrend_Volume_Plus

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.092 | +13.1% | -16.4% | 238 | FAIL |
| ETHUSDT | 0.003 | +16.5% | -17.9% | 246 | PASS |
| SOLUSDT | 0.688 | +113.3% | -23.9% | 234 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.692 | +18.8% | -9.8% | 76 | PASS |
| SOLUSDT | 0.142 | +7.5% | -12.6% | 80 | PASS |

## Code
```python
#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_12hTrend_Volume_Plus"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # 12h EMA(34) for trend filter
    ema_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Load 1d data ONCE before loop for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R3 and S3 levels
    r3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    s3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align 1d levels to 4h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume filter: > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.3 * vol_ma
    
    # ATR for volatility filter: ATR(14) > 0.5 * ATR(50)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    vol_regime = atr_14 > 0.5 * atr_50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for ATR(50)
    
    for i in range(start_idx, n):
        if np.isnan(ema_12h_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_14[i]) or np.isnan(atr_50[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R3 with 12h uptrend, volume, and volatility regime
            if (close[i] > r3_aligned[i] and close[i] > ema_12h_aligned[i] and vol_filter[i] and vol_regime[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 with 12h downtrend, volume, and volatility regime
            elif (close[i] < s3_aligned[i] and close[i] < ema_12h_aligned[i] and vol_filter[i] and vol_regime[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below S3 or trend change or low volatility
            if close[i] < s3_aligned[i] or close[i] < ema_12h_aligned[i] or not vol_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above R3 or trend change or low volatility
            if close[i] > r3_aligned[i] or close[i] > ema_12h_aligned[i] or not vol_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA(34) trend filter, volume confirmation, and volatility regime filter (ATR14 > 0.5*ATR50).
# Volatility regime filter ensures trades only occur during sufficient market movement, reducing whipsaw in low-volatility periods.
# This should improve performance in both bull and bear markets by avoiding false breakouts during consolidation.
# Position size 0.25 limits drawdown while capturing significant moves. Target: ~15-30 trades/year to avoid fee drag.
```

## Last Updated
2026-05-07 14:34

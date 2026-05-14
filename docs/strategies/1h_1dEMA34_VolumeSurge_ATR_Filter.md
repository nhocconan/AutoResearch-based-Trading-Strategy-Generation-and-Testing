# Strategy: 1h_1dEMA34_VolumeSurge_ATR_Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.601 | +57.7% | -10.6% | 162 | PASS |
| ETHUSDT | 0.161 | +28.6% | -12.6% | 203 | PASS |
| SOLUSDT | 0.861 | +146.4% | -25.6% | 206 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.390 | +1.3% | -5.1% | 77 | FAIL |
| ETHUSDT | 0.841 | +21.2% | -7.9% | 63 | PASS |
| SOLUSDT | 0.514 | +15.9% | -9.2% | 56 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h/1d momentum with volume surge and ATR volatility filter.
# Long when price breaks above 1d EMA(34) with volume > 1.8x 24-period average and ATR > 0.
# Short when price breaks below 1d EMA(34) with same conditions.
# Exit when price crosses back over 1d EMA(34).
# Uses 1d EMA for trend filter, volume surge for conviction, ATR for volatility.
# Designed for ~15-30 trades/year per symbol.
name = "1h_1dEMA34_VolumeSurge_ATR_Filter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # EMA(34) on 1d close
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # ATR(14) on 1d for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_1d)
    tr3 = np.abs(low_1d - close_1d)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume filter: current volume > 1.8 * 24-period average (24 * 1h = 1 day)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.8 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_34_1d_aligned[i]
        atr_val = atr_1d_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: price above EMA with volume surge and volatility
            if close_val > ema_val and vol_filter and atr_val > 0:
                signals[i] = 0.20
                position = 1
            # Short: price below EMA with volume surge and volatility
            elif close_val < ema_val and vol_filter and atr_val > 0:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below EMA
            if close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price crosses back above EMA
            if close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
```

## Last Updated
2026-04-18 23:41

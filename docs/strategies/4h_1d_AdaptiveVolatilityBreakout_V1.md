# Strategy: 4h_1d_AdaptiveVolatilityBreakout_V1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.525 | +46.5% | -8.7% | 480 | PASS |
| ETHUSDT | 0.217 | +31.6% | -11.3% | 488 | PASS |
| SOLUSDT | 0.375 | +50.7% | -26.0% | 470 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.156 | -4.7% | -7.2% | 162 | FAIL |
| ETHUSDT | 0.559 | +14.1% | -9.0% | 162 | PASS |
| SOLUSDT | 0.589 | +15.6% | -12.1% | 169 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_AdaptiveVolatilityBreakout_V1"
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
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) for volatility-based bands
    tr1 = np.maximum(high_1d[1:], close_1d[:-1]) - np.minimum(low_1d[1:], close_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volatility multiplier based on ATR percentile (adaptive)
    atr_series = pd.Series(atr_14)
    atr_percentile = atr_series.rolling(window=50, min_periods=20).rank(pct=True).values
    # Scale multiplier: low vol -> tighter bands, high vol -> wider bands
    vol_multiplier = 1.0 + (atr_percentile - 0.5)  # ranges from 0.5 to 1.5
    
    # Calculate adaptive bands using previous day's data
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    
    # Adaptive upper/lower bands: closer in low vol, wider in high vol
    band_width = (prev_high - prev_low) * vol_multiplier
    upper_band = prev_close + band_width * 0.5  # 50% of adaptive width above close
    lower_band = prev_close - band_width * 0.5  # 50% of adaptive width below close
    
    # Align bands to 4h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # Volume filter: current volume > 1.3x 24-period average (4h * 6 = 24h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 24)
    
    for i in range(start_idx, n):
        if np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or \
           np.isnan(vol_ma_24[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_24[i]
        
        # Volume filter
        volume_ok = vol > 1.3 * vol_ma
        
        if position == 0:
            # Long: price breaks above upper band with volume
            if price > upper_band_aligned[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band with volume
            elif price < lower_band_aligned[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns below upper band
            if price < upper_band_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns above lower band
            if price > lower_band_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-19 12:30

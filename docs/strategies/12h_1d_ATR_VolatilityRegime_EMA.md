# Strategy: 12h_1d_ATR_VolatilityRegime_EMA

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.134 | +16.3% | -7.3% | 82 | FAIL |
| ETHUSDT | 0.596 | +50.3% | -7.1% | 69 | PASS |
| SOLUSDT | 0.324 | +39.7% | -20.8% | 68 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.500 | +12.2% | -5.9% | 21 | PASS |
| SOLUSDT | 0.124 | +7.2% | -10.1% | 25 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR-based volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 14-day ATR on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR uses only high-low
    
    atr_14 = np.zeros_like(tr)
    for i in range(len(tr)):
        if i < 14:
            atr_14[i] = np.nan
        elif i == 14:
            atr_14[i] = np.mean(tr[1:15])
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # ATR percentile rank over 60 days to identify low volatility regime
    atr_percentile = np.full_like(atr_14, np.nan)
    for i in range(60, len(atr_14)):
        window = atr_14[i-60:i]
        if not np.all(np.isnan(window)):
            valid_window = window[~np.isnan(window)]
            if len(valid_window) > 0:
                atr_percentile[i] = (np.sum(valid_window <= atr_14[i]) / len(valid_window)) * 100
    
    # Align ATR percentile to 12h timeframe
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # Calculate 12-period EMA for trend direction
    ema12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # Volume confirmation: volume > 1.5x average volume (24-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=24, min_periods=24).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(60, 24)  # 60 for ATR percentile, 24 for volume avg
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_percentile_aligned[i]) or np.isnan(ema12[i]) or 
            np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        atr_percentile_val = atr_percentile_aligned[i]
        
        if position == 0:
            # Low volatility regime: ATR percentile < 30 (bottom 30% of volatility)
            # Long: price above EMA12 with volume confirmation in low vol
            if price > ema12[i] and vol > 1.5 * avg_vol[i] and atr_percentile_val < 30:
                position = 1
                signals[i] = position_size
            # Short: price below EMA12 with volume confirmation in low vol
            elif price < ema12[i] and vol > 1.5 * avg_vol[i] and atr_percentile_val < 30:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below EMA12 or volatility expands (ATR percentile > 70)
            if price < ema12[i] or atr_percentile_val > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above EMA12 or volatility expands (ATR percentile > 70)
            if price > ema12[i] or atr_percentile_val > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_ATR_VolatilityRegime_EMA"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-14 00:04

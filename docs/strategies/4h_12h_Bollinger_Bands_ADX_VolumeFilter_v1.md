# Strategy: 4h_12h_Bollinger_Bands_ADX_VolumeFilter_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.268 | +7.3% | -13.9% | 81 | FAIL |
| ETHUSDT | 0.111 | +25.1% | -17.2% | 74 | PASS |
| SOLUSDT | 0.701 | +103.5% | -26.7% | 75 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.044 | +6.0% | -7.3% | 25 | PASS |
| SOLUSDT | -0.469 | -3.0% | -18.0% | 25 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Bollinger Bands with volume confirmation and ADX trend filter.
# Long when price breaks above upper Bollinger Band on 12h timeframe, ADX > 25 (trending), and volume > 1.5x average.
# Short when price breaks below lower Bollinger Band on 12h timeframe, ADX > 25, and volume > 1.5x average.
# Exit when price returns to Bollinger middle band or ADX drops below 20 (trend weakening).
# Uses Bollinger Bands for volatility-based breakout signals, ADX for trend strength confirmation,
# and volume for institutional participation confirmation. Designed to work in both bull and bear markets
# by only trading in trending conditions (ADX > 25) and avoiding choppy markets.
# Target: 20-35 trades/year per symbol (80-140 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for Bollinger Bands and ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:  # Need enough for BB(20,2) and ADX(14)
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Bollinger Bands (20, 2)
    bb_middle = pd.Series(close_12h).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_12h).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Calculate ADX (14)
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align indicators to 4h timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, df_12h, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_12h, bb_lower)
    bb_middle_aligned = align_htf_to_ltf(prices, df_12h, bb_middle)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Volume confirmation: 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(34, 20)  # Need ADX and BB periods
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(bb_upper_aligned[i]) or 
            np.isnan(bb_lower_aligned[i]) or
            np.isnan(bb_middle_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Weak trend filter: ADX < 20 indicates trend weakening
        weak_trend = adx_aligned[i] < 20
        
        if position == 0:
            # Look for Bollinger Band breakouts in strong trend
            # Long: price breaks above upper BB AND strong trend AND volume confirmation
            if (close[i] > bb_upper_aligned[i] and 
                strong_trend and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower BB AND strong trend AND volume confirmation
            elif (close[i] < bb_lower_aligned[i] and 
                  strong_trend and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle BB or trend weakens
            if (close[i] <= bb_middle_aligned[i] or 
                weak_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to middle BB or trend weakens
            if (close[i] >= bb_middle_aligned[i] or 
                weak_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_Bollinger_Bands_ADX_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-14 08:38

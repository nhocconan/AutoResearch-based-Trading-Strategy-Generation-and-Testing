# Strategy: 4h_Bollinger_Breakout_Volume_ADX_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.249 | +33.0% | -15.7% | 226 | PASS |
| ETHUSDT | 0.049 | +20.9% | -15.6% | 222 | PASS |
| SOLUSDT | 0.599 | +83.1% | -25.8% | 199 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.503 | -8.4% | -11.0% | 76 | FAIL |
| ETHUSDT | 0.296 | +10.1% | -8.3% | 71 | PASS |
| SOLUSDT | -0.492 | -3.8% | -16.2% | 66 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Breakout with volume confirmation and ADX trend filter
# Bollinger Band breakouts capture volatility expansion and trend continuation
# Volume > 1.5x average confirms breakout strength
# ADX > 25 ensures we only trade in trending markets (works in bull and bear)
# Exit when price returns to middle band (mean reversion within trend)
# Target: 20-40 trades/year per symbol to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1h data ONCE for ADX (more responsive than 4h for trend)
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate 1h ADX (14 periods)
    adx_len = 14
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # True Range
    tr1 = high_1h[1:] - low_1h[1:]
    tr2 = np.abs(high_1h[1:] - close_1h[:-1])
    tr3 = np.abs(low_1h[1:] - close_1h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1h[1:] - high_1h[:-1]) > (low_1h[:-1] - low_1h[1:]), 
                       np.maximum(high_1h[1:] - high_1h[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.where((low_1h[:-1] - low_1h[1:]) > (high_1h[1:] - high_1h[:-1]), 
                        np.maximum(low_1h[:-1] - low_1h[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr_sum = pd.Series(tr).rolling(window=adx_len, min_periods=adx_len).sum().values
    dm_plus_sum = pd.Series(dm_plus).rolling(window=adx_len, min_periods=adx_len).sum().values
    dm_minus_sum = pd.Series(dm_minus).rolling(window=adx_len, min_periods=adx_len).sum().values
    
    # Directional Indicators
    plus_di = 100 * dm_plus_sum / tr_sum
    minus_di = 100 * dm_minus_sum / tr_sum
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=adx_len, min_periods=adx_len).mean().values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1h, adx)
    
    # Bollinger Bands (20, 2)
    bb_len = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_len, min_periods=bb_len).mean().values
    std = pd.Series(close).rolling(window=bb_len, min_periods=bb_len).std().values
    upper_band = sma + (std * bb_std)
    lower_band = sma - (std * bb_std)
    
    # Volume average (20 periods)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, bb_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(sma[i]) or 
            np.isnan(upper_band[i]) or
            np.isnan(lower_band[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long: price breaks above upper BB + volume + trend
            if (close[i] > upper_band[i-1] and 
                volume_confirmed and 
                trending):
                position = 1
                signals[i] = position_size
            # Enter short: price breaks below lower BB + volume + trend
            elif (close[i] < lower_band[i-1] and 
                  volume_confirmed and 
                  trending):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle band (mean reversion)
            if close[i] < sma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to middle band (mean reversion)
            if close[i] > sma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Bollinger_Breakout_Volume_ADX_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-14 06:47

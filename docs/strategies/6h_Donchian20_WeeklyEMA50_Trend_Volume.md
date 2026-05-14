# Strategy: 6h_Donchian20_WeeklyEMA50_Trend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.220 | +15.8% | -6.5% | 43 | FAIL |
| ETHUSDT | 0.095 | +24.2% | -9.2% | 35 | PASS |
| SOLUSDT | 0.408 | +44.8% | -13.6% | 40 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.799 | +12.2% | -5.4% | 12 | PASS |
| SOLUSDT | 0.044 | +6.4% | -4.2% | 10 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend direction
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly EMA50 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Get daily data for Donchian channel
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily Donchian(20) channel
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    donch_high_20 = high_1d_series.rolling(window=20, min_periods=20).max().values
    donch_low_20 = low_1d_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # ATR(14) for volatility filter and stop
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(donch_high_20_aligned[i]) or 
            np.isnan(donch_low_20_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above daily Donchian high with volume and above weekly EMA50
            if close[i] > donch_high_20_aligned[i] and volume_filter[i] and close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily Donchian low with volume and below weekly EMA50
            elif close[i] < donch_low_20_aligned[i] and volume_filter[i] and close[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below daily Donchian low OR ATR-based stop
            if close[i] < donch_low_20_aligned[i] or close[i] < (high[max(0, i-1)] - 1.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above daily Donchian high OR ATR-based stop
            if close[i] > donch_high_20_aligned[i] or close[i] > (low[max(0, i-1)] + 1.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-17 14:33

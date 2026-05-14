# Strategy: 6h_trix_volume_chop_regime_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.335 | +10.1% | -11.5% | 24 | FAIL |
| ETHUSDT | 0.179 | +28.3% | -11.3% | 24 | PASS |
| SOLUSDT | 0.160 | +28.3% | -17.2% | 21 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.401 | +11.2% | -7.5% | 11 | PASS |
| SOLUSDT | -2.155 | -14.3% | -16.2% | 9 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 6h_trix_volume_chop_regime_v1
# Hypothesis: 6h strategy using TRIX (triple-smoothed EMA) for momentum, volume confirmation, and chop regime filter.
# Long when TRIX crosses above zero with volume > 1.5x 20-period average and chop < 61.8 (trending).
# Short when TRIX crosses below zero with volume > 1.5x 20-period average and chop < 61.8 (trending).
# Exit when TRIX crosses back through zero in opposite direction.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 12-25 trades/year (50-100 total over 4 years) on BTC/ETH/SOL to avoid overtrading and fee drag.
# Works in both bull and bear markets: TRIX captures momentum shifts, volume confirms conviction, chop filter avoids whipsaws in ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_trix_volume_chop_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # TRIX calculation: triple-smoothed EMA of ROC
    # ROC(1) = (close[t] - close[t-1]) / close[t-1]
    close_s = pd.Series(close)
    roc = close_s.pct_change(periods=1)  # (close[t] - close[t-1]) / close[t-1]
    
    # Triple EMA smoothing: EMA(EMA(EMA(ROC)))
    ema1 = roc.ewm(span=15, min_periods=15, adjust=False).mean()
    ema2 = ema1.ewm(span=15, min_periods=15, adjust=False).mean()
    ema3 = ema2.ewm(span=15, min_periods=15, adjust=False).mean()
    trix = ema3.values * 100  # Multiply by 100 for readability
    
    # Choppiness Index regime filter (14-period)
    atr_period = 14
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr_series = pd.Series(tr)
    atr_series = tr_series.rolling(window=atr_period, min_periods=atr_period).mean()
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    highest_high = high_series.rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = low_series.rolling(window=atr_period, min_periods=atr_period).min().values
    atr_sum = tr_series.rolling(window=atr_period, min_periods=atr_period).sum().values
    chop = 100 * np.log10(atr_sum / np.log10(atr_period) / (highest_high - lowest_low))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(trix[i]) or np.isnan(trix[i-1]) or 
            np.isnan(volume_ma[i]) or np.isnan(chop[i]) or
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # Regime filter: chop < 61.8 indicates trending market
        trending_market = chop[i] < 61.8
        
        if position == 1:  # Long position
            # Exit: TRIX crosses below zero
            if trix[i] < 0 and trix[i-1] >= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TRIX crosses above zero
            if trix[i] > 0 and trix[i-1] <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for TRIX zero-cross with volume and regime confirmation
            bullish_cross = (trix[i] > 0 and trix[i-1] <= 0) and volume_confirmed and trending_market
            bearish_cross = (trix[i] < 0 and trix[i-1] >= 0) and volume_confirmed and trending_market
            
            if bullish_cross:
                position = 1
                signals[i] = 0.25
            elif bearish_cross:
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-09 01:19

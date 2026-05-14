# Strategy: 4h_TRIX_Volume_Spike_Regime_1d

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.137 | +26.2% | -9.5% | 314 | PASS |
| ETHUSDT | 0.388 | +42.0% | -9.0% | 306 | PASS |
| SOLUSDT | 0.777 | +102.4% | -18.3% | 294 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.208 | +4.0% | -5.6% | 103 | FAIL |
| ETHUSDT | 0.794 | +18.4% | -8.0% | 99 | PASS |
| SOLUSDT | 0.546 | +14.2% | -10.2% | 103 | PASS |

## Code
```python
#!/usr/bin/env python3
name = "4h_TRIX_Volume_Spike_Regime_1d"
timeframe = "4h"
leverage = 1.0

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
    
    # ===== 1d Trend Filter (HTF) =====
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) for trend
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # ===== TRIX (LTF) =====
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) - 1-period percent change
    close_s = pd.Series(close)
    ema1 = close_s.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = ema3.pct_change() * 100  # Convert to percentage
    trix = trix.fillna(0).values
    
    # ===== Volume Spike Filter =====
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    # ===== Choppiness Index (LTF) - Regime Filter =====
    # Chop = 100 * log10(sum(ATR, 14) / (max(high,14) - min(low,14))) / log10(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high - low
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range14 = max_high - min_low
    
    chop = np.full_like(close, 50.0, dtype=float)  # Default to middle
    mask = range14 != 0
    chop[mask] = 100 * np.log10(atr14[mask] / range14[mask]) / np.log10(14)
    
    # Regime: Chop > 61.8 = range (mean revert), Chop < 38.2 = trending (trend follow)
    chop_range = chop > 61.8
    chop_trend = chop < 38.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(trix[i]) or
            np.isnan(vol_spike[i]) or
            np.isnan(chop_range[i]) or np.isnan(chop_trend[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX turning up + price above 1d EMA34 + volume spike + trending regime
            if (trix[i] > trix[i-1] and  # TRIX rising
                close[i] > ema34_1d_aligned[i] and  # Price above 1d EMA34
                vol_spike[i] and
                chop_trend[i]):  # Trending regime
                signals[i] = 0.25
                position = 1
            # Short: TRIX turning down + price below 1d EMA34 + volume spike + trending regime
            elif (trix[i] < trix[i-1] and  # TRIX falling
                  close[i] < ema34_1d_aligned[i] and  # Price below 1d EMA34
                  vol_spike[i] and
                  chop_trend[i]):  # Trending regime
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX turns down OR price crosses below 1d EMA34
            if trix[i] < trix[i-1] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX turns up OR price crosses above 1d EMA34
            if trix[i] > trix[i-1] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 04:17

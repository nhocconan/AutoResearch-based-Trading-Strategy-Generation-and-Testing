# Strategy: 6h_Ichimoku_TKCross_CloudFilter_1dEMA50_Trend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.821 | -8.8% | -24.9% | 95 | FAIL |
| ETHUSDT | 0.148 | +27.2% | -18.1% | 92 | PASS |
| SOLUSDT | 0.913 | +129.5% | -19.1% | 88 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.073 | +6.4% | -8.6% | 34 | PASS |
| SOLUSDT | -0.543 | -4.5% | -13.9% | 31 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h Ichimoku Cloud Breakout with 1d Trend Filter
Hypothesis: Ichimoku cloud acts as dynamic support/resistance. TK (Tenkan-Kijun) cross above/below cloud indicates momentum shift. Using 1d EMA50 as higher-timeframe trend filter ensures alignment with daily trend, reducing false signals in choppy markets. Works in bull markets (breakouts above cloud) and bear markets (breakdowns below cloud) by requiring trend alignment. Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 12-25 trades/year on 6h.
"""

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
    
    # Get 1d data for EMA trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # The cloud is between senkou_a and senkou_b
    # Upper cloud boundary: max(senkou_a, senkou_b)
    # Lower cloud boundary: min(senkou_a, senkou_b)
    upper_cloud = np.maximum(senkou_a, senkou_b)
    lower_cloud = np.minimum(senkou_a, senkou_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Ichomoku calculations
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_tenkan = tenkan[i]
        curr_kijun = kijun[i]
        upper_cloud_level = upper_cloud[i]
        lower_cloud_level = lower_cloud[i]
        ema_trend = ema_50_1d_aligned[i]
        
        # TK cross: Tenkan crossing above/below Kijun
        tk_cross_up = (curr_tenkan > curr_kijun) and (i > start_idx) and (tenkan[i-1] <= kijun[i-1])
        tk_cross_down = (curr_tenkan < curr_kijun) and (i > start_idx) and (tenkan[i-1] >= kijun[i-1])
        
        if position == 0:
            # Look for entry signals
            # Long: TK cross above AND price above upper cloud AND price > 1d EMA50 (uptrend)
            long_entry = tk_cross_up and (curr_close > upper_cloud_level) and (curr_close > ema_trend)
            # Short: TK cross below AND price below lower cloud AND price < 1d EMA50 (downtrend)
            short_entry = tk_cross_down and (curr_close < lower_cloud_level) and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: TK cross below OR price falls below lower cloud OR price < 1d EMA50 (trend change)
            if tk_cross_down or (curr_close < lower_cloud_level) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: TK cross above OR price rises above upper cloud OR price > 1d EMA50 (trend change)
            if tk_cross_up or (curr_close > upper_cloud_level) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TKCross_CloudFilter_1dEMA50_Trend"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-25 06:19

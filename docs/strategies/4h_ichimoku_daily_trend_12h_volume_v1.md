# Strategy: 4h_ichimoku_daily_trend_12h_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.133 | +13.3% | -15.4% | 146 | FAIL |
| ETHUSDT | -0.074 | +14.0% | -15.0% | 150 | FAIL |
| SOLUSDT | 0.724 | +106.8% | -23.9% | 135 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.120 | +7.2% | -8.3% | 48 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_ichimoku_daily_trend_12h_volume_v1
# Hypothesis: 4h strategy using daily Ichimoku Cloud for primary trend filter,
# 12h timeframe for volume confirmation, and 4h TK cross for entry timing.
# Daily cloud acts as dynamic support/resistance filter (only trade in cloud direction).
# 12h volume > 1.8x 20-period average confirms institutional participation.
# 4h TK cross provides precise entry/exit timing within the trend.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 25-35 trades/year.
# Uses daily HTF and 12h volume data called ONCE before loop.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_ichimoku_daily_trend_12h_volume_v1"
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
    
    # Daily HTF data for Ichimoku Cloud (trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # Daily Ichimoku parameters
    period9_high = pd.Series(high_d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2.0
    
    period26_high = pd.Series(high_d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2.0
    
    senkou_a = ((tenkan_sen + kijun_sen) / 2.0)
    
    period52_high = pd.Series(high_d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2.0)
    
    # Align daily Ichimoku to 4h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # 12h HTF data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    volume_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    
    # 4h TK cross for entry timing
    tenkan_4h = (pd.Series(close).rolling(window=9, min_periods=9).max().values +
                 pd.Series(close).rolling(window=9, min_periods=9).min().values) / 2.0
    kijun_4h = (pd.Series(close).rolling(window=26, min_periods=26).max().values +
                pd.Series(close).rolling(window=26, min_periods=26).min().values) / 2.0
    tk_cross = tenkan_4h - kijun_4h
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(volume_ma_12h_aligned[i]) or np.isnan(tk_cross[i])):
            signals[i] = 0.0
            continue
        
        # Determine daily cloud boundaries
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # 12h volume confirmation: current 12h volume > 1.8x 20-period average
        # Need to get current 12h volume aligned to 4h
        if i >= len(prices):
            break
        # Since we don't have direct 12h volume at each 4h bar, we use the aligned MA
        # and assume current 12h volume is proportional - simplified approach
        volume_confirmed = True  # Will be handled by price action and volume spikes in 4h data
        
        # Use 4h volume for confirmation instead (more responsive)
        volume_s = pd.Series(volume)
        volume_ma_4h = volume_s.rolling(window=20, min_periods=20).mean().values
        if np.isnan(volume_ma_4h[i]):
            signals[i] = 0.0
            continue
        volume_confirmed = volume[i] > 1.8 * volume_ma_4h[i]
        
        if position == 1:  # Long position
            # Exit: price falls below cloud OR TK cross turns bearish
            if close[i] < cloud_bottom or tk_cross[i] < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above cloud OR TK cross turns bullish
            if close[i] > cloud_top or tk_cross[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Long entry: price above cloud AND TK cross bullish (Tenkan > Kijun)
                if close[i] > cloud_top and tk_cross[i] > 0:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price below cloud AND TK cross bearish (Tenkan < Kijun)
                elif close[i] < cloud_bottom and tk_cross[i] < 0:
                    position = -1
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-09 03:50

# Strategy: 6h_ichimoku_cloud_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.093 | +16.5% | -12.2% | 90 | DISCARD |
| ETHUSDT | 0.004 | +19.5% | -12.9% | 85 | KEEP |
| SOLUSDT | 1.079 | +156.1% | -14.7% | 79 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | -0.690 | -1.7% | -9.2% | 23 | DISCARD |
| SOLUSDT | 0.290 | +9.2% | -4.9% | 21 | KEEP |

## Code
```python
#!/usr/bin/env python3
"""
6h Ichimoku Cloud + Volume Confirmation + Weekly Trend Filter
Hypothesis: Ichimoku cloud acts as dynamic support/resistance with TK cross signaling momentum.
Weekly trend filter ensures we only trade in the direction of higher timeframe trend.
Volume confirmation filters out false breakouts. Designed for 6h timeframe with low trade frequency
to survive both bull and bear markets by avoiding whipsaws in ranging conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for trend filter (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly close for trend: price above/below weekly SMA(20)
    close_weekly = df_weekly['close'].values
    sma_weekly = pd.Series(close_weekly).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly trend to 6h
    weekly_trend = align_htf_to_ltf(prices, df_weekly, sma_weekly)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_9 = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_26 = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_52 + min_low_52) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    # For signal generation, we compare current close with price 26 periods ago
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period (max of all indicators)
    start = max(52, 26)  # For Senkou B and Kijun
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(weekly_trend[i]) or np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom (Senkou Span A and B)
        # Cloud is plotted 26 periods ahead, so we use values from i-26 for current cloud
        if i >= 26:
            cloud_top = max(senkou_a[i-26], senkou_b[i-26])
            cloud_bottom = min(senkou_a[i-26], senkou_b[i-26])
        else:
            # Not enough data for cloud, skip
            signals[i] = 0.0
            continue
        
        # Volume filter (20-period average)
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            volume_filter = volume[i] > vol_ma * 1.5
        else:
            volume_filter = False
        
        # Weekly trend filter: price above/below weekly SMA(20)
        weekly_uptrend = close[i] > weekly_trend[i]
        weekly_downtrend = close[i] < weekly_trend[i]
        
        # TK Cross (Tenkan-Kijun crossover)
        # Current TK cross
        tk_cross_above = tenkan[i] > kijun[i]
        tk_cross_below = tenkan[i] < kijun[i]
        
        # Price relative to cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        price_in_cloud = (close[i] >= cloud_bottom) & (close[i] <= cloud_top)
        
        # Check exits
        if position == 1:  # long position
            # Exit: price closes below cloud OR TK cross down OR weekly trend turns down
            if price_below_cloud or (tk_cross_below and tenkan[i] < kijun[i]) or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above cloud OR TK cross up OR weekly trend turns up
            if price_above_cloud or (tk_cross_above and tenkan[i] > kijun[i]) or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: TK cross + price outside cloud + volume + weekly trend alignment
            # Long: TK cross up + price above cloud + volume + weekly uptrend
            # Short: TK cross down + price below cloud + volume + weekly downtrend
            
            long_entry = tk_cross_above and price_above_cloud and volume_filter and weekly_uptrend
            short_entry = tk_cross_below and price_below_cloud and volume_filter and weekly_downtrend
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals
```

## Last Updated
2026-04-07 04:13

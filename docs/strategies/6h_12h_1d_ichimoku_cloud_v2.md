# Strategy: 6h_12h_1d_ichimoku_cloud_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.237 | +5.5% | -15.4% | 64 | FAIL |
| ETHUSDT | -0.056 | +12.1% | -17.3% | 57 | FAIL |
| SOLUSDT | 0.815 | +144.8% | -28.8% | 64 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.374 | +13.3% | -12.4% | 18 | PASS |

## Code
```python
#!/usr/bin/env python3
# 6h_12h_1d_ichimoku_cloud_v2
# Hypothesis: 6h strategy using 12h trend filter and 1d Ichimoku cloud for entries.
# Long: Price breaks above 1d Senkou Span A with volume > 1.8x 20-period average and 12h close > 12h open.
# Short: Price breaks below 1d Senkou Span B with volume > 1.8x 20-period average and 12h close < 12h open.
# Exit: Price returns to opposite Senkou Span (long exits below Span A, short exits above Span B).
# Uses 12h trend filter: only long when 12h close > 12h EMA20, only short when 12h close < 12h EMA20.
# Target: 12-30 trades/year to minimize fee drag while maintaining edge.
# Ichimoku cloud provides dynamic support/resistance that adapts to volatility, working in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_ichimoku_cloud_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_prices = prices['open'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 periods for Ichimoku
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2.0)
    
    # Align Ichimoku components to 6h (they are already shifted in calculation)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    open_12h = df_12h['open'].values
    
    # 12h EMA20 for trend filter
    close_12h_s = pd.Series(close_12h)
    ema_20_12h = close_12h_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 12h EMA20 and bullish/bearish candle to 6h
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
    open_12h_aligned = align_htf_to_ltf(prices, df_12h, open_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if any required data is NaN
        if (np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or
            np.isnan(volume[i]) or np.isnan(open_prices[i]) or
            np.isnan(ema_20_12h_aligned[i]) or np.isnan(close_12h_aligned[i]) or
            np.isnan(open_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x 20-period average
        volume_confirmed = volume[i] > 1.8 * volume_ma[i]
        # 12h bullish candle: close > open
        candle_12h_bullish = close_12h_aligned[i] > open_12h_aligned[i]
        # 12h bearish candle: close < open
        candle_12h_bearish = close_12h_aligned[i] < open_12h_aligned[i]
        # 12h trend filter: close > EMA20 for uptrend, < EMA20 for downtrend
        trend_12h_up = close_12h_aligned[i] > ema_20_12h_aligned[i]
        trend_12h_down = close_12h_aligned[i] < ema_20_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Price returns to Senkou Span A
            if close[i] <= senkou_span_a_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to Senkou Span B
            if close[i] >= senkou_span_b_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above Senkou Span A with volume, 12h bullish candle, and uptrend
            if (close[i] > senkou_span_a_aligned[i] and    # Break above Senkou Span A
                volume_confirmed and                       # Volume spike
                candle_12h_bullish and                     # 12h bullish candle
                trend_12h_up):                             # 12h uptrend
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Senkou Span B with volume, 12h bearish candle, and downtrend
            elif (close[i] < senkou_span_b_aligned[i] and  # Break below Senkou Span B
                  volume_confirmed and                     # Volume spike
                  candle_12h_bearish and                   # 12h bearish candle
                  trend_12h_down):                         # 12h downtrend
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-09 00:21

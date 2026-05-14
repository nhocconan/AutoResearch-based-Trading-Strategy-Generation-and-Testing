# Strategy: 6h_1d_1w_ichimoku_williams_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.002 | +20.6% | -11.6% | 65 | FAIL |
| ETHUSDT | 0.152 | +27.1% | -10.8% | 52 | PASS |
| SOLUSDT | 1.230 | +165.1% | -19.5% | 65 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.180 | +7.8% | -10.0% | 20 | PASS |
| SOLUSDT | -0.548 | -0.7% | -10.2% | 20 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Ichimoku cloud for trend direction and 1w Williams %R for mean reversion extremes
# - Uses 1d HTF for Ichimoku: price above/below cloud determines trend
# - Uses 1w HTF for Williams %R: extreme readings (>80 or <20) signal mean reversion opportunities
# - In bullish trend (price > cloud): look for long entries when weekly Williams %R < 20 (oversold)
# - In bearish trend (price < cloud): look for short entries when weekly Williams %R > 80 (overbought)
# - Volume confirmation: current 6h volume > 1.3x 20-period average to avoid low-volume false signals
# - Fixed position size 0.25 to control drawdown
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_1d_1w_ichimoku_williams_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # The cloud is between Senkou Span A and Senkou Span B
    # For trend determination: price above cloud = bullish, below cloud = bearish
    
    # Calculate 1w Williams %R (14 periods)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    period14_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    period14_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r = (period14_high - close_1w) / (period14_high - period14_low + 1e-10) * -100
    
    # Align all HTF data to 6h timeframe (wait for completed HTF bar)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma_20[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.3x average
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i]
        
        # Determine Ichimoku cloud boundaries
        upper_cloud = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Trend determination: price above/below cloud
        bullish_trend = close[i] > upper_cloud
        bearish_trend = close[i] < lower_cloud
        
        # Williams %R extremes: <20 = oversold, >80 = overbought
        oversold = williams_r_aligned[i] < 20
        overbought = williams_r_aligned[i] > 80
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit conditions
            if bullish_trend:
                # In bullish trend: exit when overbought or trend changes to bearish
                if overbought or bearish_trend:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            else:
                # Not in bullish trend: exit
                position = 0
                signals[i] = 0.0
                
        elif position == -1:  # Short position
            # Exit conditions
            if bearish_trend:
                # In bearish trend: exit when oversold or trend changes to bullish
                if oversold or bullish_trend:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
            else:
                # Not in bearish trend: exit
                position = 0
                signals[i] = 0.0
        else:  # Flat
            # Entry logic based on trend and Williams %R extremes
            if volume_confirmed:
                if bullish_trend and oversold:
                    # In bullish trend, weekly oversold: long mean reversion
                    position = 1
                    signals[i] = position_size
                elif bearish_trend and overbought:
                    # In bearish trend, weekly overbought: short mean reversion
                    position = -1
                    signals[i] = -position_size
    
    return signals
```

## Last Updated
2026-04-09 19:09

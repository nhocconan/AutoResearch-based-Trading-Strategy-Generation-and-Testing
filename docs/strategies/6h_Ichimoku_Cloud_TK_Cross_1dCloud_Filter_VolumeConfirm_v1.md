# Strategy: 6h_Ichimoku_Cloud_TK_Cross_1dCloud_Filter_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.452 | +0.1% | -13.4% | 115 | FAIL |
| ETHUSDT | 0.042 | +20.8% | -11.6% | 109 | PASS |
| SOLUSDT | 0.659 | +95.5% | -17.9% | 109 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.206 | +8.7% | -9.4% | 35 | PASS |
| SOLUSDT | -0.336 | -1.7% | -14.3% | 35 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h Ichimoku Cloud TK Cross with 1d Cloud Filter and Volume Confirmation
Hypothesis: Ichimoku Tenkan-Kijun (TK) cross on 6h provides timely entry signals,
while the 1d Ichimoku Cloud acts as a strong trend filter (price above cloud = bullish bias,
price below cloud = bearish bias). Volume confirmation (>1.5x 20-bar vol MA) ensures
breakout strength. This combination works in bull markets via long TK crosses above cloud
and in bear markets via short TK crosses below cloud. The cloud filter reduces whipsaws
in choppy markets and improves generalization to bear markets (2025+ test period).
Target: 50-150 total trades over 4 years = 12-37/year. Size: 0.25.
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
    
    # Get 1d data for Ichimoku Cloud (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components for 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high_52 = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Align Ichimoku components to 6h timeframe (with appropriate delay for leading spans)
    tenkan_1d = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_1d = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_1d = align_htf_to_ltf(prices, df_1d, senkou_a, additional_delay_bars=26)
    senkou_b_1d = align_htf_to_ltf(prices, df_1d, senkou_b, additional_delay_bars=26)
    
    # Calculate Ichimoku components for 6h (for TK cross)
    high_9_6h = pd.Series(close).rolling(window=9, min_periods=9).max().values
    low_9_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (high_9_6h + low_9_6h) / 2
    
    high_26_6h = pd.Series(close).rolling(window=26, min_periods=26).max().values
    low_26_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h = (high_26_6h + low_26_6h) / 2
    
    # Calculate 20-period volume MA for volume confirmation (6h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for 6h Ichimoku, 1d Ichimoku (aligned), and volume MA
    start_idx = max(26, 52)  # 26 for 6h Kijun, 52 for 1d Senkou B calculation
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_6h[i]) or 
            np.isnan(kijun_6h[i]) or 
            np.isnan(tenkan_1d[i]) or 
            np.isnan(kijun_1d[i]) or 
            np.isnan(senkou_a_1d[i]) or 
            np.isnan(senkou_b_1d[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        tenkan_6h_val = tenkan_6h[i]
        kijun_6h_val = kijun_6h[i]
        tenkan_1d_val = tenkan_1d[i]
        kijun_1d_val = kijun_1d[i]
        senkou_a_1d_val = senkou_a_1d[i]
        senkou_b_1d_val = senkou_b_1d[i]
        vol_ma = vol_ma_20[i]
        
        # Determine 1d Cloud boundaries (Senkou Span A and B)
        cloud_top = max(senkou_a_1d_val, senkou_b_1d_val)
        cloud_bottom = min(senkou_a_1d_val, senkou_b_1d_val)
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        # TK cross signals
        tk_cross_bull = tenkan_6h_val > kijun_6h_val
        tk_cross_bear = tenkan_6h_val < kijun_6h_val
        
        # Price relative to cloud
        price_above_cloud = curr_close > cloud_top
        price_below_cloud = curr_close < cloud_bottom
        
        if position == 0:
            # Long: bullish TK cross + price above 1d cloud + volume confirmation
            long_signal = tk_cross_bull and price_above_cloud and volume_confirm
            # Short: bearish TK cross + price below 1d cloud + volume confirmation
            short_signal = tk_cross_bear and price_below_cloud and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish TK cross OR price falls below cloud bottom
            if tk_cross_bear or price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish TK cross OR price rises above cloud top
            if tk_cross_bull or price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_1dCloud_Filter_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-25 03:14

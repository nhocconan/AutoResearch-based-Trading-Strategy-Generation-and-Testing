# Strategy: 4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_TrendFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.512 | +46.6% | -8.5% | 382 | PASS |
| ETHUSDT | 0.157 | +28.0% | -12.6% | 352 | PASS |
| SOLUSDT | 1.003 | +150.9% | -22.8% | 323 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.217 | +3.7% | -5.5% | 129 | FAIL |
| ETHUSDT | 0.888 | +20.5% | -7.8% | 131 | PASS |
| SOLUSDT | 0.994 | +23.2% | -8.4% | 118 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_TrendFilter
Hypothesis: Trade 4h timeframe using Camarilla R1/S1 breakouts filtered by 1d EMA34 trend and volume spikes.
Enter long when price breaks above Camarilla R1 level AND 1d trend is bullish (close > EMA34) AND volume > 1.5x 20-period average.
Enter short when price breaks below Camarilla S1 level AND 1d trend is bearish (close < EMA34) AND volume > 1.5x 20-period average.
Exit when price re-enters the Camarilla H3/L3 range or 1d trend reverses.
Uses discrete sizing 0.25 to manage risk and minimize fee churn. Target 20-50 trades/year on 4h timeframe.
Camarilla levels provide mathematically derived support/resistance that works well in ranging and trending markets.
1d EMA34 filter ensures we only trade with the higher timeframe trend, reducing counter-trend whipsaws in both bull and bear markets.
Volume spike confirmation adds conviction to breakouts, filtering out false signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels for 4h timeframe using previous day's OHLC
    # Camarilla levels are calculated from previous day's range
    # We need to get previous day's OHLC for each 4h bar
    
    # Resample to daily OHLC using actual Binance daily data from mtf_data
    # df_1d already contains actual Binance daily OHLC
    # For each 4h bar, we use the previous completed daily bar's OHLC
    
    # Calculate typical Camarilla levels from previous day's OHLC
    # R1 = Close + 1.1*(High-Low)/12
    # S1 = Close - 1.1*(High-Low)/12
    # H3 = Close + 1.1*(High-Low)/6
    # L3 = Close - 1.1*(High-Low)/6
    
    prev_day_high = df_1d['high'].shift(1).values  # Previous day's high
    prev_day_low = df_1d['low'].shift(1).values    # Previous day's low
    prev_day_close = df_1d['close'].shift(1).values # Previous day's close
    
    # Calculate Camarilla levels based on previous day's OHLC
    camarilla_range = prev_day_high - prev_day_low
    r1 = prev_day_close + 1.1 * camarilla_range / 12
    s1 = prev_day_close - 1.1 * camarilla_range / 12
    h3 = prev_day_close + 1.1 * camarilla_range / 6
    l3 = prev_day_close - 1.1 * camarilla_range / 6
    
    # Align Camarilla levels to 4h timeframe (previous day's levels are valid for entire next day)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for daily EMA34 (34) and volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND daily trend bullish (close > EMA34) AND volume spike
            long_setup = (close[i] > r1_aligned[i]) and \
                         (close[i] > ema_34_1d_aligned[i]) and \
                         volume_spike[i]
            # Short: price breaks below S1 AND daily trend bearish (close < EMA34) AND volume spike
            short_setup = (close[i] < s1_aligned[i]) and \
                          (close[i] < ema_34_1d_aligned[i]) and \
                          volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price re-enters Camarilla H3/L3 range (closes below H3 AND above L3) OR daily trend turns bearish
            if (close[i] < h3_aligned[i] and close[i] > l3_aligned[i]) or \
               (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters Camarilla H3/L3 range (closes below H3 AND above L3) OR daily trend turns bullish
            if (close[i] < h3_aligned[i] and close[i] > l3_aligned[i]) or \
               (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_TrendFilter"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 13:33

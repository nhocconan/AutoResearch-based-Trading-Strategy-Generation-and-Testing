# Strategy: 6h_Donchian20_12hBreakout_1dTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.197 | +30.5% | -15.1% | 50 | PASS |
| ETHUSDT | 0.158 | +28.3% | -15.4% | 51 | PASS |
| SOLUSDT | 0.654 | +103.9% | -22.4% | 47 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.789 | -3.8% | -8.8% | 20 | FAIL |
| ETHUSDT | 0.313 | +11.1% | -13.0% | 19 | PASS |
| SOLUSDT | 0.147 | +7.6% | -14.2% | 15 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for multi-timeframe analysis
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high_12h = np.full(len(high_12h), np.nan)
    donchian_low_12h = np.full(len(low_12h), np.nan)
    for i in range(19, len(high_12h)):
        donchian_high_12h[i] = np.max(high_12h[i-19:i+1])
        donchian_low_12h[i] = np.min(low_12h[i-19:i+1])
    
    # Align 12h Donchian to 6h timeframe
    donchian_high_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_high_12h)
    donchian_low_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_low_12h)
    
    # Calculate 1d EMA (34-period) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: volume > 2.0 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20), EMA (34), volume MA (20)
    start_idx = max(20, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_12h_aligned[i]) or np.isnan(donchian_low_12h_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: significant volume spike
        vol_filter = vol_now > 2.0 * vol_avg
        
        # Trend filter from 1d EMA
        bullish_trend = price > ema_34_aligned[i]
        bearish_trend = price < ema_34_aligned[i]
        
        upper_band = donchian_high_12h_aligned[i]
        lower_band = donchian_low_12h_aligned[i]
        
        if position == 0:
            # Long: price breaks above 12h Donchian high + volume + bullish 1d trend
            if price > upper_band and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: price breaks below 12h Donchian low + volume + bearish 1d trend
            elif price < lower_band and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below 12h Donchian low or trend turns bearish
            if price < lower_band or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above 12h Donchian high or trend turns bullish
            if price > upper_band or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_12hBreakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-27 10:11

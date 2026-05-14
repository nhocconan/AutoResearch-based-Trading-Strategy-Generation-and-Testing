# Strategy: 6h_1d_Donchian_Volume_EMA200Trend_EMA50Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.410 | +37.2% | -6.9% | 36 | KEEP |
| ETHUSDT | -0.102 | +15.2% | -14.8% | 37 | DISCARD |
| SOLUSDT | 0.877 | +107.1% | -20.8% | 30 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.027 | +6.2% | -5.9% | 13 | KEEP |
| SOLUSDT | -0.766 | -5.7% | -18.7% | 14 | DISCARD |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Donchian channels (20-period) - use previous bar's high/low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # 6h average volume (20-period) - previous bar
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    # 6h EMA200 trend filter
    ema_200_6h = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # 6h ATR (14-period) for stop-loss
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().shift(1).values
    
    # 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d EMA200 for trend filter (HTF)
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    start = max(20, 200, 14)
    for i in range(start, n):
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(avg_vol[i]) or np.isnan(ema_200_6h[i]) or np.isnan(atr[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: breakout above upper band + volume confirmation + price above EMA200_6h + EMA50_1d > EMA200_1d (bullish)
            if (price > upper[i] and vol > 2.0 * avg_vol[i] and 
                price > ema_200_6h[i] and ema_50_1d_aligned[i] > ema_200_1d_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: breakout below lower band + volume confirmation + price below EMA200_6h + EMA50_1d < EMA200_1d (bearish)
            elif (price < lower[i] and vol > 2.0 * avg_vol[i] and 
                  price < ema_200_6h[i] and ema_50_1d_aligned[i] < ema_200_1d_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below lower band OR below EMA200_6h OR stop-loss hit
            if (price < lower[i] or price < ema_200_6h[i] or 
                price < entry_price_long - 2.0 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above upper band OR above EMA200_6h OR stop-loss hit
            if (price > upper[i] or price > ema_200_6h[i] or 
                price > entry_price_short + 2.0 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        
        # Track entry price for stop-loss calculation
        if position != 0 and signals[i] != 0 and (i == start or signals[i-1] == 0):
            if position == 1:
                entry_price_long = close[i]
            else:
                entry_price_short = close[i]
    
    return signals

name = "6h_1d_Donchian_Volume_EMA200Trend_EMA50Filter"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-14 00:01

# Strategy: 6h_Williams_Fractal_Breakout_EMA_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.008 | +17.0% | -14.1% | 107 | FAIL |
| ETHUSDT | 0.160 | +28.6% | -14.2% | 118 | PASS |
| SOLUSDT | 0.719 | +129.1% | -31.5% | 122 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.376 | +13.0% | -10.5% | 38 | PASS |
| SOLUSDT | 0.133 | +7.2% | -16.3% | 43 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with daily trend filter and volume confirmation
# In trending markets, fractal breaks signal acceleration; in ranging markets, filters reduce false signals
# Works in bull/bear by using daily EMA trend filter and requiring volume confirmation
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and fractals
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA(50) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Fractals on daily data
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n] > high[n+1] and high[n] > high[n+2]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n] < low[n+1] and low[n] < low[n+2]
    n_1d = len(high_1d)
    bearish_fractal = np.zeros(n_1d, dtype=bool)
    bullish_fractal = np.zeros(n_1d, dtype=bool)
    
    for i in range(2, n_1d - 2):
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i] > high_1d[i-1] and 
            high_1d[i] > high_1d[i+1] and 
            high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = True
            
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i] < low_1d[i-1] and 
            low_1d[i] < low_1d[i+1] and 
            low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = True
    
    # Convert to price levels (use the fractal high/low as breakout level)
    bearish_fractal_level = np.where(bearish_fractal, high_1d, np.nan)
    bullish_fractal_level = np.where(bullish_fractal, low_1d, np.nan)
    
    # Forward fill to get the most recent fractal level
    bearish_fractal_level = pd.Series(bearish_fractal_level).ffill().values
    bullish_fractal_level = pd.Series(bullish_fractal_level).ffill().values
    
    # Align fractal levels to 6h timeframe with 2-bar delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal_level, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal_level, additional_delay_bars=2)
    
    # Volume confirmation: volume > 1.3x average volume (30-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=30, min_periods=30).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 30)  # for 50-period EMA and 30-period volume average
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price breaks above bullish fractal level AND above daily EMA50 with volume filter
            if (price > bullish_fractal_aligned[i] and price > ema_50_1d_aligned[i] and 
                vol > 1.3 * avg_vol[i]):
                position = 1
                signals[i] = position_size
            # Short: price breaks below bearish fractal level AND below daily EMA50 with volume filter
            elif (price < bearish_fractal_aligned[i] and price < ema_50_1d_aligned[i] and 
                  vol > 1.3 * avg_vol[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below bullish fractal level OR below daily EMA50
            if price < bullish_fractal_aligned[i] or price < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above bearish fractal level OR above daily EMA50
            if price > bearish_fractal_aligned[i] or price > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Williams_Fractal_Breakout_EMA_Volume"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-14 00:49

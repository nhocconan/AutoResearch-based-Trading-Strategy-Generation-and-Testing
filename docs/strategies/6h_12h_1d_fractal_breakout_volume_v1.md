# Strategy: 6h_12h_1d_fractal_breakout_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.279 | +4.8% | -14.0% | 94 | FAIL |
| ETHUSDT | 0.093 | +23.5% | -13.7% | 85 | PASS |
| SOLUSDT | 0.996 | +183.6% | -26.4% | 83 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.103 | +6.9% | -12.9% | 30 | PASS |
| SOLUSDT | -0.601 | -7.0% | -17.8% | 29 | FAIL |

## Code
```python
# 6h_12h_1d_fractal_breakout_volume_v1
# Hypothesis: Combine 12h trend (EMA21) with 1d Williams fractal breakouts and volume confirmation.
# Williams fractals identify key support/resistance levels. Breakouts in direction of 12h trend with volume
# capture institutional moves. Works in bull/bear via trend filter. Target: 15-30 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_fractal_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h EMA(21) for trend
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Get 1d data for Williams fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Williams fractals: need 5 bars (2 left, center, 2 right)
    high_1d = df_12h['high'].values if False else df_1d['high'].values  # Fix: use 1d data
    low_1d = df_1d['low'].values
    
    # Bearish fractal: high[n] is highest of [n-2, n-1, n, n+1, n+2]
    bearish = np.zeros(len(high_1d), dtype=bool)
    for i in range(2, len(high_1d)-2):
        if (high_1d[i] > high_1d[i-2] and high_1d[i] > high_1d[i-1] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish[i] = True
    
    # Bullish fractal: low[n] is lowest of [n-2, n-1, n, n+1, n+2]
    bullish = np.zeros(len(low_1d), dtype=bool)
    for i in range(2, len(low_1d)-2):
        if (low_1d[i] < low_1d[i-2] and low_1d[i] < low_1d[i-1] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish[i] = True
    
    # Convert to price levels (use the fractal high/low as resistance/support)
    bearish_level = np.where(bearish, high_1d, np.nan)
    bullish_level = np.where(bullish, low_1d, np.nan)
    
    # Forward fill to get the most recent fractal level
    bearish_series = pd.Series(bearish_level)
    bullish_series = pd.Series(bullish_level)
    bearish_ffilled = bearish_series.ffill().values
    bullish_ffilled = bullish_series.ffill().values
    
    # Align to 6h with 2-bar delay for fractal confirmation (needs 2 future 1d bars)
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_ffilled, additional_delay_bars=2)
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_ffilled, additional_delay_bars=2)
    
    # Volume confirmation: volume > 1.5x average of last 24 periods (24*6h = 6 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(bearish_aligned[i]) or np.isnan(bullish_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to bullish fractal support or trend changes
            if close[i] <= bullish_aligned[i] or ema_12h_aligned[i] < ema_12h_aligned[max(0, i-1)]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price returns to bearish fractal resistance or trend changes
            if close[i] >= bearish_aligned[i] or ema_12h_aligned[i] > ema_12h_aligned[max(0, i-1)]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above bearish fractal resistance with volume and 12h uptrend
            if (not np.isnan(bearish_aligned[i]) and close[i] > bearish_aligned[i] and 
                ema_12h_aligned[i] > ema_12h_aligned[max(0, i-1)] and  # Uptrend confirmation
                vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below bullish fractal support with volume and 12h downtrend
            elif (not np.isnan(bullish_aligned[i]) and close[i] < bullish_aligned[i] and 
                  ema_12h_aligned[i] < ema_12h_aligned[max(0, i-1)] and  # Downtrend confirmation
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-08 10:49

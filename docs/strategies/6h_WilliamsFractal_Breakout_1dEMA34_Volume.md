# Strategy: 6h_WilliamsFractal_Breakout_1dEMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.336 | +36.1% | -12.0% | 55 | PASS |
| ETHUSDT | 0.399 | +42.6% | -11.8% | 47 | PASS |
| SOLUSDT | 1.091 | +167.1% | -18.4% | 48 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.081 | -1.3% | -6.5% | 18 | FAIL |
| ETHUSDT | 0.392 | +10.6% | -6.6% | 12 | PASS |
| SOLUSDT | -0.208 | +2.9% | -7.1% | 12 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1d EMA34 trend filter and volume confirmation
# Uses 1d EMA34 for trend direction, weekly Williams fractal breakouts for entry signals,
# and volume spikes (2x 20-period average) to confirm breakouts. Works in both bull and bear
# markets by following the 1d trend while entering on fractal breakouts. Target: 15-25 trades/year
# to minimize fee decay while capturing trend continuation moves. Focus on BTC/ETH.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for Williams fractals
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d for trend
    close_1d = df_1d['close'].values
    ema_len = 34
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_len:
        multiplier = 2 / (ema_len + 1)
        ema_1d[ema_len-1] = np.mean(close_1d[:ema_len])
        for i in range(ema_len, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    # Calculate Williams fractals on 1w
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    n_1w = len(high_1w)
    bearish_fractal = np.zeros(n_1w)  # High fractal (sell signal)
    bullish_fractal = np.zeros(n_1w)   # Low fractal (buy signal)
    
    # Williams fractal: need 5 points (2 left, center, 2 right)
    for i in range(2, n_1w - 2):
        # Bearish fractal: high[i] is highest among 5 points
        if (high_1w[i] > high_1w[i-2] and high_1w[i] > high_1w[i-1] and 
            high_1w[i] > high_1w[i+1] and high_1w[i] > high_1w[i+2]):
            bearish_fractal[i] = high_1w[i]
        
        # Bullish fractal: low[i] is lowest among 5 points
        if (low_1w[i] < low_1w[i-2] and low_1w[i] < low_1w[i-1] and 
            low_1w[i] < low_1w[i+1] and low_1w[i] < low_1w[i+2]):
            bullish_fractal[i] = low_1w[i]
    
    # Williams fractal needs 2 extra bars for confirmation after the center bar
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 20-period average volume on 6h for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(50, 5) + 20  # EMA34 needs 34, fractal needs 5 bars, vol needs 20
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume confirmation: at least 2x average volume
        volume_confirmation = vol_ratio > 2.0
        
        if position == 0:
            # Long: Bullish fractal breakout with uptrend and volume
            if price > bullish_fractal_aligned[i] and price > ema_1d_aligned[i] and volume_confirmation:
                signals[i] = size
                position = 1
            # Short: Bearish fractal breakdown with downtrend and volume
            elif price < bearish_fractal_aligned[i] and price < ema_1d_aligned[i] and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below bullish fractal level or trend reversal
            if price < bullish_fractal_aligned[i] or price < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above bearish fractal level or trend reversal
            if price > bearish_fractal_aligned[i] or price > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WilliamsFractal_Breakout_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-27 13:54

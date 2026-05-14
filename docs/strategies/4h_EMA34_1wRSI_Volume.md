# Strategy: 4h_EMA34_1wRSI_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.158 | +28.0% | -10.6% | 90 | PASS |
| ETHUSDT | 0.266 | +37.0% | -15.7% | 104 | PASS |
| SOLUSDT | 0.178 | +29.7% | -31.5% | 115 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.354 | +1.4% | -8.2% | 34 | FAIL |
| ETHUSDT | 0.354 | +11.7% | -10.9% | 27 | PASS |
| SOLUSDT | 0.326 | +11.6% | -15.6% | 27 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Price above/below 1d EMA(34) with volume confirmation and 1w RSI(14) regime filter.
In bull markets: price above 1d EMA(34) acts as dynamic support, buy on dips with volume.
In bear markets: price below 1d EMA(34) acts as resistance, sell on rallies with volume.
Weekly RSI avoids extremes: only long when RSI(1w)<60, short when RSI(1w)>40.
Designed for 15-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    ema = np.full(len(close), np.nan)
    if len(close) < period:
        return ema
    ema[period-1] = np.mean(close[:period])
    for i in range(period, len(close)):
        ema[i] = (close[i] * 2 / (period + 1)) + ema[i-1] * (1 - 2 / (period + 1))
    return ema

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    if len(close) < period + 1:
        return np.full(len(close), np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close), np.nan)
    avg_loss = np.full(len(close), np.nan)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    rs = np.full(len(close), np.nan)
    rsi = np.full(len(close), np.nan)
    
    for i in range(period, len(close)):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100
    
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA(34)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Get 1w data for RSI(14)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(34) on 1d
    ema_34_1d = calculate_ema(close_1d, 34)
    
    # Calculate RSI(14) on 1w
    rsi_14_1w = calculate_rsi(close_1w, 14)
    
    # Align to 4h timeframe
    ema_34_1d_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    rsi_14_1w_4h = align_htf_to_ltf(prices, df_1w, rsi_14_1w)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_1d_4h[i]) or np.isnan(rsi_14_1w_4h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3 * 20-period average
        vol_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Long: price above EMA(34), RSI not overbought, volume confirmation
            if close[i] > ema_34_1d_4h[i] and rsi_14_1w_4h[i] < 60 and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price below EMA(34), RSI not oversold, volume confirmation
            elif close[i] < ema_34_1d_4h[i] and rsi_14_1w_4h[i] > 40 and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below EMA(34) or RSI becomes overbought
            if close[i] <= ema_34_1d_4h[i] or rsi_14_1w_4h[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above EMA(34) or RSI becomes oversold
            if close[i] >= ema_34_1d_4h[i] or rsi_14_1w_4h[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_EMA34_1wRSI_Volume"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-18 12:18

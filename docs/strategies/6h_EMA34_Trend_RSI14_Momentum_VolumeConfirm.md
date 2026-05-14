# Strategy: 6h_EMA34_Trend_RSI14_Momentum_VolumeConfirm

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.153 | +10.0% | -15.3% | 236 | FAIL |
| ETHUSDT | 0.041 | +19.5% | -14.2% | 247 | PASS |
| SOLUSDT | 1.027 | +195.2% | -22.0% | 220 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.262 | +10.0% | -9.2% | 85 | PASS |
| SOLUSDT | -0.302 | -2.0% | -17.1% | 80 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and 6h data for entry timing
    df_1d = get_htf_data(prices, '1d')
    df_6h = get_htf_data(prices, '6h')
    
    if len(df_1d) < 34 or len(df_6h) < 20:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 6h RSI(14) for momentum filter
    close_6h = df_6h['close'].values
    delta = pd.Series(close_6h).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_6h = 100 - (100 / (1 + rs))
    rsi_6h = rsi_6h.fillna(50).values
    rsi_6h_aligned = align_htf_to_ltf(prices, df_6h, rsi_6h)
    
    # Volume confirmation: current volume > 1.3x average volume (reduced to increase trades slightly)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > vol_ma * 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(rsi_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from 1d EMA
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Momentum filter from 6h RSI
        rsi_momentum_up = rsi_6h_aligned[i] > 50
        rsi_momentum_down = rsi_6h_aligned[i] < 50
        
        # Entry conditions: require 1d trend, 6h momentum, and volume confirmation
        long_entry = uptrend and rsi_momentum_up and volume_confirm[i]
        short_entry = downtrend and rsi_momentum_down and volume_confirm[i]
        
        # Exit conditions: when trend or momentum reverses
        if position == 1:
            exit_condition = not (uptrend and rsi_momentum_up)
        elif position == -1:
            exit_condition = not (downtrend and rsi_momentum_down)
        else:
            exit_condition = False
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif exit_condition and position != 0:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_EMA34_Trend_RSI14_Momentum_VolumeConfirm"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-28 09:56

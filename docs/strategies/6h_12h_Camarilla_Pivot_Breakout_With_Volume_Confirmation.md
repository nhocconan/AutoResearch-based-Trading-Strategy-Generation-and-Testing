# Strategy: 6h_12h_Camarilla_Pivot_Breakout_With_Volume_Confirmation

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.019 | +15.5% | -21.8% | 106 | FAIL |
| ETHUSDT | 0.556 | +72.9% | -19.4% | 92 | PASS |
| SOLUSDT | 0.847 | +178.6% | -43.3% | 78 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.065 | +32.2% | -10.2% | 26 | PASS |
| SOLUSDT | 0.287 | +11.3% | -15.9% | 25 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
6h_12h_Camarilla_Pivot_Breakout_With_Volume_Confirmation
Hypothesis: Buy when price breaks above Camarilla H4 level with volume > 1.5x 20-period average and 12h EMA50 uptrend, sell when price breaks below L4 level with volume confirmation and 12h EMA50 downtrend. Uses 6h primary timeframe with 12h trend filter. Works in bull markets via upside breakouts and in bear markets via downside breaks. Target: 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate True Range and ATR for volatility
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[0], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    # Previous period's high/low for Camarilla calculation
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels for current period using previous period's range
    # H4 = close + 1.1 * (high - low) / 2
    # L4 = close - 1.1 * (high - low) / 2
    camarilla_h4 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l4 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # 12h EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        ema50_12h = np.full(len(prices), np.nan)
    else:
        close_12h = df_12h['close'].values
        ema50_12h_raw = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema50_12h = align_htf_to_ltf(prices, df_12h, ema50_12h_raw)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):  # warmup period
        # Skip if any required data is not ready
        if (np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or 
            np.isnan(ema50_12h[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long signal: break above Camarilla H4 with volume expansion and 12h uptrend
        long_signal = (close[i] > camarilla_h4[i] and 
                      volume_expansion[i] and 
                      close[i] > ema50_12h[i])
        
        # Short signal: break below Camarilla L4 with volume expansion and 12h downtrend
        short_signal = (close[i] < camarilla_l4[i] and 
                       volume_expansion[i] and 
                       close[i] < ema50_12h[i])
        
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "6h_12h_Camarilla_Pivot_Breakout_With_Volume_Confirmation"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-13 19:39

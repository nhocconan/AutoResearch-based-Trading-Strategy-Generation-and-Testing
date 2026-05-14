# Strategy: 12H_WilliamsR_1dEMA34_VolumeSpike_ATRStop

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.250 | +13.5% | -11.3% | 40 | FAIL |
| ETHUSDT | 0.025 | +20.9% | -10.3% | 39 | PASS |
| SOLUSDT | 0.902 | +99.1% | -12.6% | 26 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.197 | +8.2% | -7.3% | 13 | PASS |
| SOLUSDT | -0.599 | -0.2% | -8.8% | 7 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R reversal + 1d EMA34 trend filter + volume spike.
Long when Williams %R crosses above -20 from below AND close > 1d EMA34 AND volume > 1.8x 20-period average.
Short when Williams %R crosses below -80 from above AND close < 1d EMA34 AND volume > 1.8x 20-period average.
Exit when Williams %R crosses -50 in opposite direction or ATR stoploss hit (2.0*ATR).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 15-25 trades/year per symbol.
Williams %R captures overbought/oversold reversals, while 1d EMA34 ensures alignment with higher-timeframe trend.
Volume confirmation filters weak signals. Designed for range-bound and trending markets.
"""

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
    
    # Load 1d data for EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF EMA34 to 12h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Williams %R(14) on 12h timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(100, 14, 34, 20)  # Ensure warmup for Williams %R(14), EMA34, ATR(14), Vol MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        wr = williams_r[i]
        wr_prev = williams_r[i-1]
        
        if position == 0:
            # Long: Williams %R crosses above -20 from below AND 1d EMA34 uptrend AND volume spike
            if (wr > -20 and wr_prev <= -20 and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Williams %R crosses below -80 from above AND 1d EMA34 downtrend AND volume spike
            elif (wr < -80 and wr_prev >= -80 and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Williams %R crosses -50 in opposite direction
            if position == 1 and wr < -50 and wr_prev >= -50:
                exit_signal = True
            elif position == -1 and wr > -50 and wr_prev <= -50:
                exit_signal = True
            
            # ATR-based stoploss: 2.0 * ATR from entry
            if position == 1 and price < entry_price - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > entry_price + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsR_1dEMA34_VolumeSpike_ATRStop"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-23 05:04

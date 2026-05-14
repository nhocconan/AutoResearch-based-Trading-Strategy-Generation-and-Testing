# Strategy: 1h_EMA21_Trend_4hATRBreakout_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.514 | +3.0% | -10.5% | 554 | FAIL |
| ETHUSDT | 0.225 | +30.7% | -7.6% | 496 | PASS |
| SOLUSDT | -0.188 | +4.1% | -25.3% | 476 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.678 | +13.8% | -5.5% | 161 | PASS |

## Code
```python
#!/usr/bin/env python3

"""
Hypothesis: 1-hour EMA21 trend following with 4-hour ATR-based range filter and volume spike.
Only trade in the direction of the EMA21 trend (up or down) when price breaks above/below
the 4-hour ATR-based upper/lower bounds with volume confirmation. Uses 4h ATR to define
volatility-adjusted breakout levels, avoiding false breakouts in low volatility periods.
Designed for low trade frequency (15-35 trades/year) by requiring multiple confirmations:
trend alignment, volatility breakout, and volume spike. Works in both bull and bear markets
by following the EMA21 trend direction, which adapts to market conditions.
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
    
    # EMA21 on 1h for trend direction
    close_s = pd.Series(close)
    ema21 = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Load 4h data for ATR-based range - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Calculate ATR(14) on 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h EMA20 as middle reference
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Upper and lower bands: EMA20 ± 1.5 * ATR
    upper_band = ema20_4h + 1.5 * atr_14
    lower_band = ema20_4h - 1.5 * atr_14
    
    # Align to 1h
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    upper_band_aligned = align_htf_to_ltf(prices, df_4h, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_4h, lower_band)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema21[i]) or np.isnan(ema20_4h_aligned[i]) or 
            np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: EMA21 uptrend + price breaks above upper band + volume spike
            if ema21[i] > ema21[i-1] and close[i] > upper_band_aligned[i] and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short: EMA21 downtrend + price breaks below lower band + volume spike
            elif ema21[i] < ema21[i-1] and close[i] < lower_band_aligned[i] and vol_spike:
                signals[i] = -0.20
                position = -1
        else:
            # Exit: EMA21 trend reversal or price returns to middle band
            exit_signal = False
            
            if position == 1:
                # Exit long: EMA21 turns down or price closes below EMA20
                if ema21[i] < ema21[i-1] or close[i] < ema20_4h_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: EMA21 turns up or price closes above EMA20
                if ema21[i] > ema21[i-1] or close[i] > ema20_4h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_EMA21_Trend_4hATRBreakout_Volume"
timeframe = "1h"
leverage = 1.0
```

## Last Updated
2026-04-22 19:21

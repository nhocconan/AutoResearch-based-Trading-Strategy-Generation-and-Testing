# Strategy: 12H_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeConfirmation_L4H4Exit_ATRTrailingStop

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.091 | +24.1% | -9.2% | 74 | PASS |
| ETHUSDT | 0.018 | +19.4% | -16.8% | 69 | PASS |
| SOLUSDT | 0.324 | +47.6% | -33.2% | 67 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.216 | -7.8% | -11.2% | 32 | FAIL |
| ETHUSDT | 0.429 | +13.1% | -7.7% | 24 | PASS |
| SOLUSDT | -0.373 | -2.0% | -17.2% | 26 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above Camarilla R3 AND close > 1d EMA50 AND volume > 1.8x 20-period average.
Short when price breaks below Camarilla S3 AND close < 1d EMA50 AND volume > 1.8x 20-period average.
Exit when price retraces to Camarilla H4/L4 level or ATR trailing stop (2.0*ATR from extreme).
Uses discrete position sizing (0.28) and volume filter to target 12-37 trades/year.
12h timeframe reduces noise while maintaining sufficient trade frequency for BTC/ETH in both bull/bear regimes.
Camarilla levels provide precise intraday support/resistance derived from prior day's range.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Camarilla levels from 1d timeframe (based on prior day's OHLC)
    # Camarilla formula: 
    # H4 = close + 1.1*(high-low)*1.1/2
    # L4 = close - 1.1*(high-low)*1.1/2
    # H3 = close + 1.1*(high-low)*1.1/4
    # L3 = close - 1.1*(high-low)*1.1/4
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    rng = high_1d - low_1d
    h4 = close_1d + 1.1 * rng * 1.1 / 2.0
    l4 = close_1d - 1.1 * rng * 1.1 / 2.0
    h3 = close_1d + 1.1 * rng * 1.1 / 4.0
    l3 = close_1d - 1.1 * rng * 1.1 / 4.0
    
    # Align Camarilla levels to 12h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 needs 50, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema50_val = ema50_1d_aligned[i]
        h3 = h3_aligned[i]
        l3 = l3_aligned[i]
        h4 = h4_aligned[i]
        l4 = l4_aligned[i]
        
        if position == 0:
            # Long: Break above Camarilla H3 AND uptrend (price > EMA50) AND volume spike (1.8x avg)
            if close[i] > h3 and close[i] > ema50_val and volume[i] > 1.8 * vol_ma_val:
                signals[i] = 0.28
                position = 1
                highest_since_entry = price
            # Short: Break below Camarilla L3 AND downtrend (price < EMA50) AND volume spike (1.8x avg)
            elif close[i] < l3 and close[i] < ema50_val and volume[i] > 1.8 * vol_ma_val:
                signals[i] = -0.28
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retraces to Camarilla L4 (for long) or H4 (for short)
            if position == 1 and close[i] <= l4:
                exit_signal = True
            elif position == -1 and close[i] >= h4:
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.28 if position == 1 else -0.28
    
    return signals

name = "12H_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeConfirmation_L4H4Exit_ATRTrailingStop"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-23 07:37

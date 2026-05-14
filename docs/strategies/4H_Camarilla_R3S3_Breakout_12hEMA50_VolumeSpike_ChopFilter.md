# Strategy: 4H_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike_ChopFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.399 | +35.9% | -8.9% | 228 | KEEP |
| ETHUSDT | 0.260 | +31.8% | -7.8% | 217 | KEEP |
| SOLUSDT | 0.485 | +54.8% | -15.9% | 180 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.095 | -8.3% | -8.6% | 107 | DISCARD |
| ETHUSDT | 0.858 | +16.4% | -10.3% | 88 | KEEP |
| SOLUSDT | 0.580 | +12.7% | -6.4% | 65 | KEEP |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout + 12h EMA50 trend + volume spike + choppiness regime filter.
Long when price breaks above Camarilla R3 AND close > 12h EMA50 AND volume > 2.0x 20-period average AND chop < 61.8 (trending).
Short when price breaks below Camarilla S3 AND close < 12h EMA50 AND volume > 2.0x 20-period average AND chop < 61.8.
Exit when price crosses Camarilla H3/L3 levels or ATR stoploss (2.5x ATR).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 20-40 trades/year per symbol.
Adds 12h trend filter and choppiness regime to avoid ranging markets and improve BTC/ETH performance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for price action - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ATR(14) on 4h data for stoploss
    tr1 = np.maximum(high_4h - low_4h, np.abs(high_4h - np.roll(close_4h, 1)))
    tr2 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high_4h[0] - low_4h[0]  # first bar
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume average (20-period) on 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data for Camarilla levels - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous 1d bar
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # First bar has no previous data
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    range_1d = prev_high - prev_low
    camarilla_r3 = prev_close + 1.1 * range_1d / 4
    camarilla_s3 = prev_close - 1.1 * range_1d / 4
    camarilla_h3 = prev_close + 1.1 * range_1d / 2
    camarilla_l3 = prev_close - 1.1 * range_1d / 2
    
    # Load 12h data for EMA50 and choppiness filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate EMA50 on 12h data
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Choppiness Index on 12h data (period=14)
    atr_12h_list = []
    for i in range(len(high_12h)):
        tr = max(high_12h[i] - low_12h[i], abs(high_12h[i] - close_12h[i-1]) if i > 0 else 0, abs(low_12h[i] - close_12h[i-1]) if i > 0 else 0)
        atr_12h_list.append(tr)
    atr_12h = pd.Series(atr_12h_list).rolling(window=14, min_periods=14).mean().values
    
    max_high_12h = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    min_low_12h = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_sum_12h = atr_12h * 14
    max_min_diff_12h = max_high_12h - min_low_12h
    chop_12h = np.where(
        (range_sum_12h > 0) & (max_min_diff_12h > 0),
        100 * np.log10(range_sum_12h / max_min_diff_12h) / np.log10(14),
        50.0  # neutral when undefined
    )
    
    # Align 1d indicators to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Align 12h indicators to 4h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(chop_12h_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND close > 12h EMA50 AND volume spike AND trending market (chop < 61.8)
            if (price > camarilla_r3_aligned[i] and 
                close[i] > ema50_12h_aligned[i] and 
                volume[i] > 2.0 * vol_ma_val and
                chop_12h_aligned[i] < 61.8):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Camarilla S3 AND close < 12h EMA50 AND volume spike AND trending market (chop < 61.8)
            elif (price < camarilla_s3_aligned[i] and 
                  close[i] < ema50_12h_aligned[i] and 
                  volume[i] > 2.0 * vol_ma_val and
                  chop_12h_aligned[i] < 61.8):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below Camarilla H3 or ATR stoploss
                if price < camarilla_h3_aligned[i]:
                    exit_signal = True
                elif price < entry_price - 2.5 * atr_4h[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above Camarilla L3 or ATR stoploss
                if price > camarilla_l3_aligned[i]:
                    exit_signal = True
                elif price > entry_price + 2.5 * atr_4h[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-23 04:26

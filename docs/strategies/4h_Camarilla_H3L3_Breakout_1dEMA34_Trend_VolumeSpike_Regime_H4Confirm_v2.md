# Strategy: 4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_Regime_H4Confirm_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.263 | +28.7% | -5.4% | 151 | PASS |
| ETHUSDT | 0.077 | +23.6% | -6.3% | 145 | PASS |
| SOLUSDT | 0.614 | +60.4% | -13.9% | 141 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.104 | +0.1% | -2.6% | 58 | FAIL |
| ETHUSDT | 1.151 | +17.5% | -4.4% | 46 | PASS |
| SOLUSDT | 0.802 | +13.5% | -4.0% | 43 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_Regime_H4Confirm_v2
Hypothesis: Tighten entry conditions by requiring volume spike AND price closing above/below Camarilla H3/L3 for two consecutive 4h bars to reduce false breakouts and lower trade frequency. Uses 1d EMA34 trend filter and 4h HMA21 for additional confirmation. Targets 15-25 trades/year on 4h timeframe to minimize fee drag while capturing strong trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels (H3, L3) from previous day
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: H3 = close + (high - low) * 1.1/4, L3 = close - (high - low) * 1.1/4
    camarilla_h3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_l3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (completed 1d bar only)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 4h HMA21 for additional trend confirmation
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # HMA calculation: WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    wma_half = pd.Series(close_4h).rolling(window=half_len, min_periods=half_len).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True
    ).values
    wma_full = pd.Series(close_4h).rolling(window=21, min_periods=21).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True
    ).values
    raw_hma = 2 * wma_half - wma_full
    hma_21 = pd.Series(raw_hma).rolling(window=sqrt_len, min_periods=sqrt_len).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True
    ).values
    hma_21_aligned = align_htf_to_ltf(prices, df_4h, hma_21)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Two-bar confirmation: price closed above/below level in previous bar
    close_prev = np.roll(close, 1)
    close_prev[0] = np.nan
    camarilla_h3_prev = np.roll(camarilla_h3_aligned, 1)
    camarilla_h3_prev[0] = np.nan
    camarilla_l3_prev = np.roll(camarilla_l3_aligned, 1)
    camarilla_l3_prev[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 1d Camarilla, 1d EMA34, 4h HMA21, volume MA, and two-bar confirmation
    start_idx = max(1, 34, 21, 20, 2)  # +2 for two-bar confirmation
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(hma_21_aligned[i]) or 
            np.isnan(vol_ma[i]) or
            np.isnan(close_prev[i]) or
            np.isnan(camarilla_h3_prev[i]) or
            np.isnan(camarilla_l3_prev[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price closes above H3 for two consecutive bars + 1d uptrend + 4h HMA21 uptrend + volume spike
            long_setup = (close[i] > camarilla_h3_aligned[i]) and \
                         (close_prev[i] > camarilla_h3_prev[i]) and \
                         (close[i] > ema_34_1d_aligned[i]) and \
                         (close[i] > hma_21_aligned[i]) and \
                         volume_spike[i]
            # Short: price closes below L3 for two consecutive bars + 1d downtrend + 4h HMA21 downtrend + volume spike
            short_setup = (close[i] < camarilla_l3_aligned[i]) and \
                          (close_prev[i] < camarilla_l3_prev[i]) and \
                          (close[i] < ema_34_1d_aligned[i]) and \
                          (close[i] < hma_21_aligned[i]) and \
                          volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price closes below L3 OR 1d trend turns down OR 4h HMA21 turns down
            if (close[i] < camarilla_l3_aligned[i]) or \
               (close[i] < ema_34_1d_aligned[i]) or \
               (close[i] < hma_21_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price closes above H3 OR 1d trend turns up OR 4h HMA21 turns up
            if (close[i] > camarilla_h3_aligned[i]) or \
               (close[i] > ema_34_1d_aligned[i]) or \
               (close[i] > hma_21_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_Regime_H4Confirm_v2"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-25 12:42

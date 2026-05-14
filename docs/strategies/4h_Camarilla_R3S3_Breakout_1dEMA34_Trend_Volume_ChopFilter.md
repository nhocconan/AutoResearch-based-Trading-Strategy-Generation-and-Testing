# Strategy: 4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_Volume_ChopFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.047 | +20.2% | -7.5% | 213 | FAIL |
| ETHUSDT | 0.318 | +32.6% | -7.3% | 196 | PASS |
| SOLUSDT | -0.044 | +17.3% | -11.9% | 157 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.111 | +17.5% | -3.4% | 78 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_Volume_ChopFilter
Hypothesis: On 4h timeframe, Camarilla R3/S3 breakouts from the previous 4h bar with 1d EMA34 trend filter, volume spike (>2.0x 20-bar avg), and choppiness regime filter (CHOP > 50) captures institutional breakouts with controlled trade frequency. The 4h timeframe targets 20-50 trades/year (80-200 over 4 years), balancing signal quality and execution precision. Trend alignment ensures directional bias in both bull and bear markets, volume confirms participation, chop filter avoids whipsaws in ranging markets, and discrete sizing (0.25) minimizes fee churn. Works in bull markets via long breakouts and bear markets via short breakouts.
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
    
    # Get 1d data for HTF trend and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA34 on 1d for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Choppiness Index on 1d (14-period)
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (max(high) - min(low))))
    atr_1d = np.maximum(np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1])), np.abs(low_1d[1:] - close_1d[:-1]))
    atr_1d = np.concatenate([[np.nan], atr_1d])
    atr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr_sum / (np.log10(14) * (max_high - min_low)))
    chop_1d = np.where((max_high - min_low) > 0, chop_raw, 50)  # avoid division by zero
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Get 4h data for Camarilla levels (HTF relative to 4h)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels from previous 4h bar (R3, S3)
    # Camarilla: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    # Use previous completed 4h bar to avoid look-ahead
    prev_close = np.concatenate([[np.nan], close_4h[:-1]])
    prev_high = np.concatenate([[np.nan], high_4h[:-1]])
    prev_low = np.concatenate([[np.nan], low_4h[:-1]])
    
    camarilla_range = prev_high - prev_low
    r3 = prev_close + 1.1 * camarilla_range * 1.1 / 4
    s3 = prev_close - 1.1 * camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3)
    
    # Volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(50, 20, 34)  # EMA34, vol MA, chop
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_val = ema_34_aligned[i]
        chop_val = chop_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current volume > 2.0x 20-period average
        volume_spike = vol_val > 2.0 * vol_ma_val
        # Choppiness regime: CHOP > 50 indicates ranging market (avoid breakouts in strong trends)
        chop_regime = chop_val > 50
        
        if position == 0:
            # Look for entry signals: Camarilla R3/S3 breakout with trend, volume, and chop filter
            # Long: price breaks above R3 with uptrend (close > EMA34), volume spike, and chop > 50
            long_signal = (high_val > r3_val) and (close_val > ema_val) and volume_spike and chop_regime
            # Short: price breaks below S3 with downtrend (close < EMA34), volume spike, and chop > 50
            short_signal = (low_val < s3_val) and (close_val < ema_val) and volume_spike and chop_regime
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks below S3 (exit long)
            if close_val < s3_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks above R3 (exit short)
            if close_val > r3_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_Volume_ChopFilter"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-26 00:04

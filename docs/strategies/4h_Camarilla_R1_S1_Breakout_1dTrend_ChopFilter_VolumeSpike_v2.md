# Strategy: 4h_Camarilla_R1_S1_Breakout_1dTrend_ChopFilter_VolumeSpike_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.077 | +23.6% | -16.3% | 316 | PASS |
| ETHUSDT | 0.452 | +44.6% | -8.3% | 280 | PASS |
| SOLUSDT | 0.468 | +59.5% | -14.7% | 241 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.134 | +4.7% | -6.3% | 117 | FAIL |
| ETHUSDT | 1.594 | +30.9% | -7.1% | 105 | PASS |
| SOLUSDT | 0.094 | +6.8% | -13.8% | 84 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_ChopFilter_VolumeSpike_v2
Hypothesis: Tighten entry conditions from experiment #90320 to reduce trade frequency while maintaining edge.
Changes: Increased volume confirmation to 2.0x (from 1.8x), raised chop threshold to 55.0 (from 61.8) for stronger trend filter.
Target: 150-250 total 4h trades over 4 years (~38-63/year) to minimize fee drag while preserving Sharpe > 0.284 from baseline.
Works in bull/bear: 1d EMA34 trend filter adapts to market direction, chop filter avoids whipsaws in ranging markets.
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
    
    # Get 1d data for EMA34 trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Previous 1d bar's high, low, close for Camarilla levels
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels: R1, S1
    camarilla_range = prev_high - prev_low
    R1 = prev_close + camarilla_range * 1.0/12
    S1 = prev_close - camarilla_range * 1.0/12
    
    # Align Camarilla levels to 4h timeframe (1d -> 4h)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: 2.0x average volume (tighter than 1.8x)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index regime filter (14-period)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr_14 * 14 / (max_high_14 - min_low_14 + 1e-10)) / np.log10(14)
    chop_raw = np.where((max_high_14 - min_low_14) <= 0, 100, chop_raw)
    chop_raw = np.where(np.isnan(chop_raw) | np.isinf(chop_raw), 50, chop_raw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Warmup: max of 1d EMA (34), volume MA (20), ATR (14), CHOP (14)
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(chop_raw[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        ema_34_1d_val = ema_34_1d_aligned[i]
        R1_val = R1_aligned[i]
        S1_val = S1_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        chop_val = chop_raw[i]
        
        # Regime filter: only trade when not too choppy (CHOP < 55.0 = strong trending market)
        regime_filter = chop_val < 55.0
        
        if position == 0:
            # Long: break above R1, uptrend (close > 1d EMA34), volume spike, good regime
            long_signal = (high_val > R1_val) and (close_val > ema_34_1d_val) and (volume_val > 2.0 * vol_ma_val) and regime_filter
            # Short: break below S1, downtrend (close < 1d EMA34), volume spike, good regime
            short_signal = (low_val < S1_val) and (close_val < ema_34_1d_val) and (volume_val > 2.0 * vol_ma_val) and regime_filter
            
            if long_signal:
                signals[i] = 0.30
                position = 1
                entry_price = close_val
                long_stop = entry_price - 2.0 * atr_val
            elif short_signal:
                signals[i] = -0.30
                position = -1
                entry_price = close_val
                short_stop = entry_price + 2.0 * atr_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            # Update trailing stop: move stop up as price makes new highs
            long_stop = max(long_stop, high_val - 2.0 * atr_val)
            # Exit: trailing stop hit or trend reversal (price < 1d EMA34) or regime becomes too choppy
            if (low_val < long_stop) or (close_val < ema_34_1d_val) or (chop_val >= 55.0):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, low_val + 2.0 * atr_val)
            # Exit: trailing stop hit or trend reversal (price > 1d EMA34) or regime becomes too choppy
            if (high_val > short_stop) or (close_val > ema_34_1d_val) or (chop_val >= 55.0):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_ChopFilter_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-26 01:28

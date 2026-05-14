# Strategy: 4h_Camarilla_R1_S1_Breakout_1dEMA34_RegimeFilter_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.069 | +23.3% | -10.1% | 196 | PASS |
| ETHUSDT | 0.192 | +28.9% | -9.3% | 173 | PASS |
| SOLUSDT | 1.016 | +120.2% | -13.7% | 160 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.875 | -0.7% | -4.9% | 71 | FAIL |
| ETHUSDT | 1.033 | +19.6% | -6.2% | 63 | PASS |
| SOLUSDT | -0.258 | +1.9% | -13.2% | 60 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_RegimeFilter_v1
Hypothesis: Trade Camarilla R1/S1 breakouts with 1d EMA34 trend and volume confirmation, filtered by choppiness regime (CHOP > 61.8 = range, CHOP < 38.2 = trend). Uses ATR trailing stop (2.0x) and requires price >1.0% from EMA34 to avoid chop. Position size 0.25. Designed for stable performance in both bull and bear markets via confluence: pivot break + HTF trend + volume spike + regime filter.
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
    
    # Get 1d data for HTF trend filter and Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels from previous 1d bar
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align 1d EMA and 1d Camarilla levels to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: 2.0x median volume (balanced for frequency)
    vol_median = pd.Series(volume).rolling(window=50, min_periods=50).median().values
    
    # ATR for stop (14-period on 4h)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index (14-period on 4h) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high, lookback) - min(low, lookback))) / log10(lookback)
    atr_14_chop = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14_chop).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denominator = np.maximum(max_high_14 - min_low_14, 1e-10)  # avoid division by zero
    chop_raw = 100 * np.log10(sum_atr_14 / chop_denominator) / np.log10(14)
    chop = np.where(chop_denominator > 0, chop_raw, 50.0)  # default to 50 when no range
    
    # Price distance from EMA34 to avoid chop (>1.0%)
    ema_distance = np.abs((close - ema_34_1d_aligned) / ema_34_1d_aligned * 100)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    bars_since_entry = 0
    
    # Warmup: max of 1d EMA (34), volume median (50), 4h ATR (14), chop (14), distance calc
    start_idx = max(34, 50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_median[i]) or 
            np.isnan(atr_14[i]) or
            np.isnan(chop[i]) or
            np.isnan(ema_distance[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_34_1d_val = ema_34_1d_aligned[i]
        camarilla_r1_val = camarilla_r1_aligned[i]
        camarilla_s1_val = camarilla_s1_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        atr_14_val = atr_14[i]
        chop_val = chop[i]
        ema_distance_val = ema_distance[i]
        
        # Regime filter: only trade in trending markets (CHOP < 38.2) or strong range (CHOP > 61.8)
        # For breakout strategy, we prefer trending markets
        regime_filter = chop_val < 38.2  # trending regime
        
        if position == 0:
            # Long: break above R1, uptrend (close > EMA34), volume spike, price >1.0% from EMA, trending regime
            long_signal = (high_val > camarilla_r1_val) and \
                          (close_val > ema_34_1d_val) and \
                          (volume_val > 2.0 * vol_median_val) and \
                          (ema_distance_val > 1.0) and \
                          regime_filter
            # Short: break below S1, downtrend (close < EMA34), volume spike, price >1.0% from EMA, trending regime
            short_signal = (low_val < camarilla_s1_val) and \
                           (close_val < ema_34_1d_val) and \
                           (volume_val > 2.0 * vol_median_val) and \
                           (ema_distance_val > 1.0) and \
                           regime_filter
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                long_stop = entry_price - 2.0 * atr_14_val
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                short_stop = entry_price + 2.0 * atr_14_val
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long with minimum holding period
            bars_since_entry += 1
            signals[i] = 0.25
            # Update trailing stop: move stop up as price makes new highs
            long_stop = max(long_stop, high_val - 2.0 * atr_14_val)
            # Exit: trailing stop hit or trend reversal (close < EMA34) after minimum holding period
            if bars_since_entry >= 3 and ((low_val < long_stop) or (close_val < ema_34_1d_val)):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short with minimum holding period
            bars_since_entry += 1
            signals[i] = -0.25
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, low_val + 2.0 * atr_14_val)
            # Exit: trailing stop hit or trend reversal (close > EMA34) after minimum holding period
            if bars_since_entry >= 3 and ((high_val > short_stop) or (close_val > ema_34_1d_val)):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_RegimeFilter_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-26 01:53

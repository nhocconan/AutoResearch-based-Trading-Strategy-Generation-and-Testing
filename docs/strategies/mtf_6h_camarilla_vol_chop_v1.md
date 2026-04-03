# Strategy: mtf_6h_camarilla_vol_chop_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.806 | -23.3% | -37.8% | 7 | FAIL |
| ETHUSDT | -0.929 | -32.7% | -38.3% | 12 | FAIL |
| SOLUSDT | 2.360 | +881.9% | -22.1% | 353 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 1.531 | +40.1% | -9.3% | 116 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #051: 6h Camarilla Pivot + 1d Volume Spike + Chop Regime Filter

HYPOTHESIS: Camarilla pivot levels on 6h act as intraday support/resistance. 
Entries occur at R3/S3 (mean reversion) or R4/S4 (breakout) only when 1d volume 
spikes (>2x 20-period average) indicate institutional participation. Chop regime 
filter (Choppiness Index > 61.8) ensures we only mean revert in ranging markets 
and breakout in trending markets, adapting to both bull and bear conditions.
Target: 75-150 trades over 4 years (19-37/year) with discrete sizing (0.25).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_camarilla_vol_chop_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume spike confirmation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === HTF: 1d data for chop regime filter (Call ONCE before loop) ===
    # Calculate Choppiness Index on 1d
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr1 = high_1d - low_1d
        tr2 = np.abs(high_1d - np.roll(close_1d, 1))
        tr3 = np.abs(low_1d - np.roll(close_1d, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high_1d[0] - low_1d[0]  # First period
        
        # Sum of TR over 14 periods
        tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
        
        # Max high and min low over 14 periods
        max_h = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
        min_l = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
        
        # Choppiness Index: 100 * log10(sum(tr)/(max(high)-min(low))) / log10(14)
        range_hl = max_h - min_l
        chop = np.zeros(len(close_1d))
        mask = (range_hl > 0) & (~np.isnan(tr_sum)) & (~np.isnan(range_hl))
        chop[mask] = 100 * np.log10(tr_sum[mask] / range_hl[mask]) / np.log10(14)
        chop[:] = np.where(np.isnan(chop), 50, chop)  # Neutral fallback
        
        chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    else:
        chop_aligned = np.full(n, 50.0)  # Neutral
    
    # === 6h Camarilla Pivot Levels (based on previous day) ===
    # Calculate from previous 1d bar (aligned and shifted)
    if len(df_1d) >= 2:
        prev_close = df_1d['close'].shift(1).values
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        
        # Camarilla calculations
        range_prev = prev_high - prev_low
        camarilla_h5 = prev_close + range_prev * 1.1 / 2  # R4
        camarilla_h4 = prev_close + range_prev * 1.1 / 4  # R3
        camarilla_h3 = prev_close + range_prev * 1.1 / 6  # R2
        camarilla_l3 = prev_close - range_prev * 1.1 / 6  # S2
        camarilla_l4 = prev_close - range_prev * 1.1 / 4  # S3
        camarilla_l5 = prev_close - range_prev * 1.1 / 2  # S4
        
        # Align to 6h and shift by 1 day (use previous day's levels)
        h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
        h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
        h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
        l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
        l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
        l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    else:
        h5_aligned = h4_aligned = h3_aligned = l3_aligned = l4_aligned = l5_aligned = np.full(n, np.nan)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(h5_aligned[i]) or np.isnan(h4_aligned[i]) or np.isnan(h3_aligned[i]) or
            np.isnan(l3_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(l5_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Determine market regime from 1d chop
        is_ranging = chop_aligned[i] > 61.8  # Chop > 61.8 = ranging (mean revert)
        is_trending = chop_aligned[i] <= 61.8  # Chop <= 61.8 = trending (breakout)
        
        # Volume confirmation: 1d volume spike > 2.0
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
        # Mean reversion in ranging markets: fade at R3/S3
        if is_ranging and volume_spike:
            # Short at R3 (h4) with stop above R4 (h5)
            if close[i] > h4_aligned[i] and close[i] <= h5_aligned[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            # Long at S3 (l4) with stop below S4 (l5)
            elif close[i] < l4_aligned[i] and close[i] >= l5_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
        
        # Breakout in trending markets: break R4/S4
        elif is_trending and volume_spike:
            # Long breakout above R4 (h5)
            if close[i] > h5_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short breakdown below S4 (l5)
            elif close[i] < l5_aligned[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
        
        # No signal
        else:
            signals[i] = 0.0
    
    return signals
```

## Last Updated
2026-04-03 10:40

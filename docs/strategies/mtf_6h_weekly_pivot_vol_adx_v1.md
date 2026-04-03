# Strategy: mtf_6h_weekly_pivot_vol_adx_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -3.279 | -18.8% | -18.8% | 43 | FAIL |
| ETHUSDT | 1.429 | +166.4% | -8.8% | 344 | PASS |
| SOLUSDT | 2.360 | +881.9% | -22.1% | 353 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.929 | +47.8% | -8.2% | 111 | PASS |
| SOLUSDT | 1.531 | +40.1% | -9.3% | 116 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #359: 6h Weekly Pivot + 12h Volume Spike + 1d ADX Trend Filter

HYPOTHESIS: Weekly pivot levels (calculated from prior week's OHLC) provide significant 
support/resistance that institutional players respect. Combining these with 12h volume 
spikes (>2.0x average) confirms institutional participation, while 1d ADX > 25 ensures 
we only trade in trending markets to avoid whipsaws. Mean reversion at weekly S1/R1 
in the trend direction, with breakouts at S2/R2 on volume spikes. Targets 15-25 
trades/year on 6h timeframe (60-100 total over 4 years) for minimal fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_weekly_pivot_vol_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for volume spike (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate volume ratio (current vs 20-period average) on 12h
    if len(df_12h) >= 20:
        vol_12h = df_12h['volume'].values
        vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_12h = np.zeros(len(vol_12h))
        vol_ratio_12h[20:] = vol_12h[20:] / vol_ma_20[20:]
        vol_ratio_12h[:20] = 1.0  # Neutral for warmup
        vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    else:
        vol_ratio_12h_aligned = np.full(n, 1.0)
    
    # === HTF: 1d data for ADX trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX(14) on 1d
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr1 = high_1d[1:] - low_1d[1:]
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                           np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
        dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                            np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed values
        atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # DI+ and DI-
        di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
        di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
        adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
        adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    else:
        adx_aligned = np.full(n, 20.0)  # Default to ranging if insufficient data
    
    # === HTF: 1w data for weekly pivot (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot levels for each 6h bar
    weekly_pp = np.full(n, np.nan)
    weekly_r1 = np.full(n, np.nan)
    weekly_s1 = np.full(n, np.nan)
    weekly_r2 = np.full(n, np.nan)
    weekly_s2 = np.full(n, np.nan)
    
    for i in range(n):
        current_time = prices.iloc[i]['open_time']
        # Find the most recent completed weekly bar before current 6h bar
        prior_weekly_bars = df_1w[df_1w['open_time'] < current_time]
        if len(prior_weekly_bars) > 0:
            prev_week = prior_weekly_bars.iloc[-1]
            ph = prev_week['high']
            pl = prev_week['low']
            pc = prev_week['close']
            
            # Standard pivot formulas
            pp = (ph + pl + pc) / 3
            r1 = 2 * pp - pl
            s1 = 2 * pp - ph
            r2 = pp + (ph - pl)
            s2 = pp - (ph - pl)
            
            weekly_pp[i] = pp
            weekly_r1[i] = r1
            weekly_s1[i] = s1
            weekly_r2[i] = r2
            weekly_s2[i] = s2
        else:
            # Not enough prior data
            weekly_pp[i] = np.nan
            weekly_r1[i] = np.nan
            weekly_s1[i] = np.nan
            weekly_r2[i] = np.nan
            weekly_s2[i] = np.nan
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(weekly_pp[i]) or np.isnan(weekly_r1[i]) or np.isnan(weekly_s1[i]) or
            np.isnan(weekly_r2[i]) or np.isnan(weekly_s2[i]) or
            np.isnan(vol_ratio_12h_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade when ADX > 25 (trending market) ---
        is_trending = adx_aligned[i] > 25
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_12h_aligned[i] > 2.0
        
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
                # Take profit at weekly S2 (strong support) or R2 (strong resistance)
                if close[i] >= weekly_r2[i] or close[i] <= weekly_s2[i]:
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
                # Take profit at weekly R2 (strong resistance) or S2 (strong support)
                if close[i] >= weekly_r2[i] or close[i] <= weekly_s2[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price at S1 (mean reversion) OR break above R1 with volume
        long_condition = is_trending and (
            (close[i] <= weekly_s1[i] * 1.001 and close[i] >= weekly_pp[i] * 0.999) or  # S1-PP mean reversion
            (close[i] > weekly_r1[i] and volume_spike)  # Breakout above R1 with volume
        )
        
        # Short: Price at R1 (mean reversion) OR break below S1 with volume
        short_condition = is_trending and (
            (close[i] >= weekly_r1[i] * 0.999 and close[i] <= weekly_pp[i] * 1.001) or  # R1-PP mean reversion
            (close[i] < weekly_s1[i] and volume_spike)  # Breakdown below S1 with volume
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals
```

## Last Updated
2026-04-03 09:34

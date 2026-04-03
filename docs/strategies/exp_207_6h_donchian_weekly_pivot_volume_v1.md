# Strategy: exp_207_6h_donchian_weekly_pivot_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.007 | +20.6% | -9.2% | 107 | PASS |
| ETHUSDT | 0.408 | +43.0% | -10.4% | 91 | PASS |
| SOLUSDT | 0.821 | +111.9% | -20.1% | 75 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.446 | +12.5% | -7.5% | 34 | PASS |
| SOLUSDT | -0.061 | +4.5% | -8.4% | 29 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #207: 6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Spike

HYPOTHESIS: Donchian channel breakouts on 6h timeframe, filtered by weekly pivot direction 
(price > weekly pivot = bullish bias, price < weekly pivot = bearish bias) and confirmed 
by volume spikes (>2.0x average), capture strong momentum moves with reduced false breakouts. 
Weekly pivot provides structural support/resistance from higher timeframe, aligning with 
institutional order flow. 6h timeframe targets 12-37 trades/year (50-150 total over 4 years) 
to minimize fee drag while capturing significant moves. Volume confirmation filters out 
low-conviction breakouts. ATR-based stoploss manages risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_207_6h_donchian_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points from 1d data (using prior week's OHLC)
    # Weekly pivot = (Prior Week High + Prior Week Low + Prior Week Close) / 3
    weekly_pivot = np.full(n, np.nan)
    
    if len(df_1d) >= 5:  # Need at least a week of data
        # Resample 1d to weekly using actual weekly boundaries
        df_1d_indexed = df_1d.copy()
        df_1d_indexed.index = pd.date_range(
            start=df_1d['open_time'].iloc[0], 
            periods=len(df_1d), 
            freq='1d'
        )
        weekly_ohlc = df_1d_indexed[['high', 'low', 'close']].resample('W').agg({
            'high': 'max',
            'low': 'min', 
            'close': 'last'
        })
        
        if len(weekly_ohlc) >= 2:  # Need at least 2 weeks (prior + current)
            # Shift by 1 to use prior week's data only (no look-ahead)
            prior_week_high = weekly_ohlc['high'].shift(1).values
            prior_week_low = weekly_ohlc['low'].shift(1).values
            prior_week_close = weekly_ohlc['close'].shift(1).values
            
            # Calculate weekly pivot for each prior week
            prior_week_pivot = (prior_week_high + prior_week_low + prior_week_close) / 3.0
            
            # Map back to daily frequency then to 6h
            # Create series aligned with 1d index
            daily_pivot = pd.Series(index=df_1d_indexed.index, dtype=np.float64)
            for i in range(1, len(weekly_ohlc)):
                week_start = weekly_ohlc.index[i]
                week_end = weekly_ohlc.index[i+1] if i+1 < len(weekly_ohlc) else weekly_ohlc.index[-1] + pd.Timedelta(days=7)
                daily_pivot.loc[week_start:week_end] = prior_week_pivot[i]
            
            # Align to 6h timeframe
            daily_pivot_values = daily_pivot.reindex(df_1d_indexed.index).values
            weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, daily_pivot_values)
        else:
            weekly_pivot_aligned = np.full(n, np.nan)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20) ===
    donchian_h = np.full(n, np.nan)
    donchian_l = np.full(n, np.nan)
    donchian_m = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_h[i] = np.max(high[i-20:i])
        donchian_l[i] = np.min(low[i-20:i])
        donchian_m[i] = (donchian_h[i] + donchian_l[i]) / 2
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # Ensure enough data for HTF weekly pivot and ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Weekly Pivot Filter: Price > pivot = bullish bias, Price < pivot = bearish bias ---
        price_above_weekly_pivot = close[i] > weekly_pivot_aligned[i]
        price_below_weekly_pivot = close[i] < weekly_pivot_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Donchian Breakout Conditions ---
        breakout_up = close[i] > donchian_h[i]
        breakout_down = close[i] < donchian_l[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian middle line reversion (take profit)
                if close[i] < donchian_m[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian middle line reversion (take profit)
                if close[i] > donchian_m[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout up + volume spike + price above weekly pivot
        long_condition = breakout_up and volume_spike and price_above_weekly_pivot
        
        # Short: Donchian breakout down + volume spike + price below weekly pivot
        short_condition = breakout_down and volume_spike and price_below_weekly_pivot
        
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
2026-04-03 11:31

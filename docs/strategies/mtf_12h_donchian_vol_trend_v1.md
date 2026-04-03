# Strategy: mtf_12h_donchian_vol_trend_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 2.360 | +881.9% | -22.1% | 353 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 1.531 | +40.1% | -9.3% | 116 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #382: 12h Donchian Breakout + 1d Volume Spike + 1w Trend Filter

HYPOTHESIS: 12h Donchian(20) breakouts combined with 1d volume confirmation (>1.8x average) 
and 1w trend filter (price > EMA50 on weekly) captures strong momentum moves while avoiding 
choppy markets. Donchian breakouts work in both bull (breakouts to new highs) and bear 
(breakdowns to new lows) markets. Higher timeframes (1d/1w) filter noise and reduce overtrading.
Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume spike (Call ONCE before loop) ===
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
    
    # === HTF: 1w data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate EMA(50) on 1w close
    if len(df_1w) >= 50:
        close_1w = df_1w['close'].values
        ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    else:
        ema_50_1w_aligned = np.full(n, np.nan)
    
    # === 12h Indicators: Calculate Donchian channels (20-period) ===
    # Need to map each 12h bar to the prior 12h bar's OHLC for Donchian calculation
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    
    # Pre-compute prior 12h OHLC for each 12h bar
    for i in range(n):
        current_time = prices.iloc[i]['open_time']
        # Find the most recent completed 12h bar before current 12h bar
        df_12h_temp = prices[prices['open_time'] < current_time].tail(1)
        if len(df_12h_temp) > 0:
            # We need to get actual 12h data - simpler approach: use rolling window on 12h data
            # Since we're on 12h timeframe, we can calculate directly
            pass
    
    # Simpler: calculate Donchian directly on 12h prices (since timeframe=12h)
    if n >= 20:
        # Calculate rolling max/min for Donchian channels
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
        donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
        # For warmup period, fill with NaN
        donchian_upper[:19] = np.nan
        donchian_lower[:19] = np.nan
    
    # === Session filter: 00-23 UTC (trade all hours for 12h timeframe) ===
    # For 12h timeframe, we can trade all hours as each bar represents 12 hours
    hours = prices.index.hour  # Pre-compute before loop
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Session Filter: Trade all hours for 12h timeframe ---
        hour = hours[i]
        # No session filter for 12h - trade continuously
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
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
                # Take profit at Donchian upper (trailing stop for longs)
                if close[i] <= donchian_lower[i]:
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
                # Take profit at Donchian lower (trailing stop for shorts)
                if close[i] >= donchian_upper[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian upper with volume confirmation and uptrend
        long_condition = (
            close[i] > donchian_upper[i] and  # Breakout above upper channel
            vol_ratio_1d_aligned[i] > 1.8 and  # Volume spike confirmation
            close[i] > ema_50_1w_aligned[i]   # Price above weekly EMA50 (uptrend)
        )
        
        # Short: Price breaks below Donchian lower with volume confirmation and downtrend
        short_condition = (
            close[i] < donchian_lower[i] and  # Breakdown below lower channel
            vol_ratio_1d_aligned[i] > 1.8 and  # Volume spike confirmation
            close[i] < ema_50_1w_aligned[i]   # Price below weekly EMA50 (downtrend)
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
2026-04-03 09:47

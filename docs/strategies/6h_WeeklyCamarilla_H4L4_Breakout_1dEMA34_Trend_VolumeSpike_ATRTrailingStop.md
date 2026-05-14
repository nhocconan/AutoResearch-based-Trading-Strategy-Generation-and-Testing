# Strategy: 6h_WeeklyCamarilla_H4L4_Breakout_1dEMA34_Trend_VolumeSpike_ATRTrailingStop

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.278 | +32.6% | -9.4% | 99 | PASS |
| ETHUSDT | 0.131 | +26.2% | -16.4% | 85 | PASS |
| SOLUSDT | 1.248 | +193.2% | -15.8% | 79 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.552 | +2.9% | -4.7% | 29 | FAIL |
| ETHUSDT | 0.357 | +9.9% | -7.3% | 26 | PASS |
| SOLUSDT | -0.346 | +2.7% | -5.9% | 22 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h Weekly Camarilla Pivot Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Weekly Camarilla levels (R4/S4) provide strong monthly support/resistance.
Breakouts with volume spike and 1d EMA34 trend alignment capture institutional flow.
Designed for 50-150 trades over 4 years. Works in bull/bear via 1d EMA34 trend filter.
Uses weekly pivot calculated from prior week's OHLC (loaded as 1w HTF).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average with min_periods"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for EMA34 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 trend filter
    ema_34_1d = calculate_ema(df_1d['close'].values, 34)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1w data for weekly Camarilla pivots (using previous week)
    df_1w = get_htf_data(prices, '1w')
    prev_weekly_high = np.roll(df_1w['high'].values, 1)
    prev_weekly_low = np.roll(df_1w['low'].values, 1)
    prev_weekly_close = np.roll(df_1w['close'].values, 1)
    prev_weekly_high[0] = np.nan
    prev_weekly_low[0] = np.nan
    prev_weekly_close[0] = np.nan
    
    prev_weekly_range = prev_weekly_high - prev_weekly_low
    # Weekly Camarilla levels
    weekly_h4 = prev_weekly_close + 1.1 * prev_weekly_range / 2  # R4 equivalent
    weekly_l4 = prev_weekly_close - 1.1 * prev_weekly_range / 2  # S4 equivalent
    weekly_h3 = prev_weekly_close + 1.1 * prev_weekly_range / 4  # R3
    weekly_l3 = prev_weekly_close - 1.1 * prev_weekly_range / 4  # S3
    weekly_h3_aligned = align_htf_to_ltf(prices, df_1w, weekly_h3, additional_delay_bars=0)
    weekly_l3_aligned = align_htf_to_ltf(prices, df_1w, weekly_l3, additional_delay_bars=0)
    weekly_h4_aligned = align_htf_to_ltf(prices, df_1w, weekly_h4, additional_delay_bars=0)
    weekly_l4_aligned = align_htf_to_ltf(prices, df_1w, weekly_l4, additional_delay_bars=0)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ATR for trailing stop (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    # Start index: need enough for EMA, volume MA, ATR
    start_idx = max(34, 20, 14) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(weekly_h3_aligned[i]) or np.isnan(weekly_l3_aligned[i]) or
            np.isnan(weekly_h4_aligned[i]) or np.isnan(weekly_l4_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Breakout conditions using weekly levels
        breakout_long = curr_close > weekly_h4_aligned[i]  # Break above weekly R4
        breakout_short = curr_close < weekly_l4_aligned[i]  # Break below weekly S4
        
        if position == 0:
            # Look for entry signals - require: Weekly Camarilla breakout + volume spike + 1d EMA34 trend alignment
            long_entry = breakout_long and vol_spike and (curr_close > ema_34_1d_aligned[i])
            short_entry = breakout_short and vol_spike and (curr_close < ema_34_1d_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_high_since_entry = curr_high
                lowest_low_since_entry = curr_low
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                highest_high_since_entry = curr_high
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: update highest high and check exit conditions
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            
            # Exit conditions: retrace to weekly L3, trend change, or ATR trailing stop
            trailing_stop = highest_high_since_entry - 2.5 * atr[i]
            if curr_close < weekly_l3_aligned[i] or curr_close < ema_34_1d_aligned[i] or curr_close < trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: update lowest low and check exit conditions
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            
            # Exit conditions: retrace to weekly H3, trend change, or ATR trailing stop
            trailing_stop = lowest_low_since_entry + 2.5 * atr[i]
            if curr_close > weekly_h3_aligned[i] or curr_close > ema_34_1d_aligned[i] or curr_close > trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyCamarilla_H4L4_Breakout_1dEMA34_Trend_VolumeSpike_ATRTrailingStop"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-25 08:33

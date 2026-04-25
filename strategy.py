#!/usr/bin/env python3
"""
12h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike
Hypothesis: On 12h timeframe, use Camarilla pivot levels (H3/L3 for mean reversion, H4/L4 for breakout) with 1d EMA34 trend filter and volume confirmation (>1.5x 20-bar average). In ranging markets (price between H3/L3), fade extremes; in trending markets (price beyond H3/L3), breakout in trend direction. Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag. Works in both bull and bear markets by adapting to price action relative to pivot levels and trend.
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
    
    # 1d data for Camarilla pivots and EMA34 (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Camarilla pivot levels (based on previous day's OHLC)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    H4 = prev_close + 1.5 * prev_range
    L4 = prev_close - 1.5 * prev_range
    H3 = prev_close + 1.125 * prev_range
    L3 = prev_close - 1.125 * prev_range
    
    # Align 1d pivot levels to 12h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start index: need enough for 1d indicators (34 for EMA, 20 for vol MA)
    start_idx = 34
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data not ready
        if (np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or np.isnan(H3_aligned[i]) or 
            np.isnan(L3_aligned[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals
            # Long signals: price above H3 and above EMA34 (uptrend) OR price below L3 and above EMA34 (mean reversion in uptrend)
            long_breakout = (curr_close > H3_aligned[i]) and (curr_close > ema_34_aligned[i])
            long_mean_revert = (curr_close < L3_aligned[i]) and (curr_close > ema_34_aligned[i])
            
            # Short signals: price below L3 and below EMA34 (downtrend) OR price above H3 and below EMA34 (mean reversion in downtrend)
            short_breakout = (curr_close < L3_aligned[i]) and (curr_close < ema_34_aligned[i])
            short_mean_revert = (curr_close > H3_aligned[i]) and (curr_close < ema_34_aligned[i])
            
            long_entry = (long_breakout or long_mean_revert) and volume_spike[i]
            short_entry = (short_breakout or short_mean_revert) and volume_spike[i]
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                bars_since_entry = 0
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price crosses below EMA34 or reaches opposite extreme (L4 for stop)
            if curr_close < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif curr_close < L4_aligned[i]:  # stoploss at L4
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price crosses above EMA34 or reaches opposite extreme (H4 for stop)
            if curr_close > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif curr_close > H4_aligned[i]:  # stoploss at H4
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0
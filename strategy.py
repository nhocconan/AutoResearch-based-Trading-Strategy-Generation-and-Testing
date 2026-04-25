#!/usr/bin/env python3
"""
1d_Camarilla_H4L4_Breakout_1wEMA34_TrendFilter_VolumeSpike
Hypothesis: Daily Camarilla H4/L4 breakout with weekly EMA34 trend filter and volume spike confirmation.
Uses ATR-based trailing stoploss for risk management. Weekly EMA34 provides strong long-term trend filter
that works in both bull and bear markets by ensuring we only trade with the dominant weekly trend.
Volume spike confirms institutional participation. Designed for 7-25 trades/year (30-100 over 4 years)
to minimize fee drag. H4/L4 levels are stronger support/resistance than H3/L3, reducing false breakouts.
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
    
    # Weekly data for EMA34 trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA34 trend filter (loaded ONCE)
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily data for Camarilla calculation (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Prior 1d bar OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: H4, L4 (stronger intraday support/resistance than H3/L3)
    camarilla_range = prev_high - prev_low
    h4 = prev_close + camarilla_range * 1.1 / 2
    l4 = prev_close - camarilla_range * 1.1 / 2
    
    # Align Camarilla levels to 1d timeframe (completed 1d bar)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ATR for volatility-based stoploss (14-period) on 1d data
    tr1_1d = high[1:] - low[1:]
    tr2_1d = np.abs(high[1:] - close[:-1])
    tr3_1d = np.abs(low[1:] - close[:-1])
    tr_1d = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for 1w EMA (34), 1d Camarilla, volume MA (20), ATR (14)
    start_idx = max(34, 20, 14)  # EMA34 + vol MA20 + ATR14
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_1d[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla H4/L4 breakout + volume spike + 1w EMA34 trend alignment
            long_breakout = curr_high > h4_aligned[i]
            short_breakout = curr_low < l4_aligned[i]
            
            # Trend filter: price must be on correct side of 1w EMA34
            long_trend = curr_close > ema_34_1w_aligned[i]
            short_trend = curr_close < ema_34_1w_aligned[i]
            
            long_entry = (long_breakout and volume_spike[i] and long_trend)
            short_entry = (short_breakout and volume_spike[i] and short_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: track highest price for trailing stop
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit when price closes below Camarilla H4 (failed breakout) or trend reverses or ATR stop hit
            atr_stop = highest_since_entry - 2.5 * atr_1d[i]
            if curr_close < h4_aligned[i] or curr_close < ema_34_1w_aligned[i] or curr_close < atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: track lowest price for trailing stop
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit when price closes above Camarilla L4 (failed breakout) or trend reverses or ATR stop hit
            atr_stop = lowest_since_entry + 2.5 * atr_1d[i]
            if curr_close > l4_aligned[i] or curr_close > ema_34_1w_aligned[i] or curr_close > atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_H4L4_Breakout_1wEMA34_TrendFilter_VolumeSpike"
timeframe = "1d"
leverage = 1.0
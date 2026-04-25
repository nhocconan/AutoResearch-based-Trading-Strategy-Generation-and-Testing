#!/usr/bin/env python3
"""
12h_Camarilla_H3L3_Breakout_1dTrend_VolumeSpike
Hypothesis: On 12h timeframe, Camarilla H3/L3 levels from 1d act as strong support/resistance. A break above H3 with volume spike and 1d uptrend (price > EMA34) signals long; break below L3 with volume spike and 1d downtrend (price < EMA34) signals short. Uses ATR-based trailing stoploss to limit drawdown. Designed for fewer trades (~20-40/year) to minimize fee drag and work in both bull and bear markets by capturing strong breakouts from key institutional levels with trend and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla levels and trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: H3 = close + (high-low)*1.1/4, L3 = close - (high-low)*1.1/4
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    camarilla_h3 = c_1d + (h_1d - l_1d) * 1.1 / 4
    camarilla_l3 = c_1d - (h_1d - l_1d) * 1.1 / 4
    
    # 1d EMA34 for trend filter (loaded ONCE)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to LTF (12h)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 12h ATR for volatility and stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 12h volume spike: current volume > 2.0 * 20-period volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    max_high = 0.0     # track highest high since entry for trailing stop (long)
    min_low = 0.0      # track lowest low since entry for trailing stop (short)
    
    # Start index: need ATR (14), volume MA (20) + aligned HTF arrays
    start_idx = max(20, 14, 0)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Long: price breaks above camarilla H3 with volume spike and 1d uptrend
            long_breakout = (curr_close > camarilla_h3_aligned[i]) and vol_spike[i] and (curr_close > ema_34_1d_aligned[i])
            # Short: price breaks below camarilla L3 with volume spike and 1d downtrend
            short_breakout = (curr_close < camarilla_l3_aligned[i]) and vol_spike[i] and (curr_close < ema_34_1d_aligned[i])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                max_high = curr_high
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                min_low = curr_low
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            max_high = max(max_high, curr_high)
            # Exit: price breaks below camarilla L3 OR trend turns down OR ATR trailing stop hit
            trailing_stop = curr_high < (max_high - 2.5 * atr_14[i])
            if (curr_close < camarilla_l3_aligned[i]) or (curr_close < ema_34_1d_aligned[i]) or trailing_stop:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            min_low = min(min_low, curr_low)
            # Exit: price breaks above camarilla H3 OR trend turns up OR ATR trailing stop hit
            trailing_stop = curr_low > (min_low + 2.5 * atr_14[i])
            if (curr_close > camarilla_h3_aligned[i]) or (curr_close > ema_34_1d_aligned[i]) or trailing_stop:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0
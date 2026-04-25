#!/usr/bin/env python3
"""
12h Camarilla H3/L3 Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Camarilla H3/L3 levels from 1d chart provide institutional breakout levels.
Breakouts in direction of 1d EMA34 trend with volume confirmation and choppy market filter
capture strong moves while avoiding whipsaws in ranging markets. Designed for 12h
timeframe targeting 12-37 trades/year. Uses discrete position sizing (0.25) to minimize fee churn.
Works in bull markets via breakout continuation and in bear markets via mean-reversion from extreme
levels when 1d trend aligns and chop filter is favorable. Uses proper MTF loading with get_htf_data called once before loop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and EMA34 (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (H3, L3) from 1d OHLC
    # Camarilla: H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    camarilla_h3 = daily_close + 1.1 * (daily_high - daily_low) / 4
    camarilla_l3 = daily_close - 1.1 * (daily_high - daily_low) / 4
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss
    if len(close) >= 14:
        tr1 = np.abs(np.diff(close, prepend=close[0]))
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr2[0] = np.abs(high[0] - close[0])
        tr3[0] = np.abs(low[0] - close[0])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.zeros(n)
        atr[:13] = np.nan
        for i in range(13, n):
            atr[i] = np.mean(tr[i-13:i+1])
    else:
        atr = np.full(n, np.nan)
    
    # Calculate 20-period volume MA for volume spike detection
    vol_ma_20 = np.full(n, np.nan)
    for i in range(n):
        start_idx = max(0, i - 19)
        vol_ma_20[i] = np.mean(volume[start_idx:i+1])
    
    # Calculate Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(atr14) / (max(high) - min(low))) / log10(14)
    if len(close) >= 14:
        # True Range components
        tr1 = np.abs(np.diff(close, prepend=close[0]))
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr2[0] = np.abs(high[0] - close[0])
        tr3[0] = np.abs(low[0] - close[0])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Sum of TR over 14 periods
        tr_sum_14 = np.full(n, np.nan)
        for i in range(n):
            start_idx = max(0, i - 13)
            tr_sum_14[i] = np.sum(tr[start_idx:i+1])
        
        # Max high and min low over 14 periods
        max_high_14 = np.full(n, np.nan)
        min_low_14 = np.full(n, np.nan)
        for i in range(n):
            start_idx = max(0, i - 13)
            max_high_14[i] = np.max(high[start_idx:i+1])
            min_low_14[i] = np.min(low[start_idx:i+1])
        
        # Choppiness Index
        chop = np.full(n, np.nan)
        for i in range(13, n):
            if max_high_14[i] > min_low_14[i] and tr_sum_14[i] > 0:
                chop[i] = 100 * np.log10(tr_sum_14[i] / (max_high_14[i] - min_low_14[i])) / np.log10(14)
            else:
                chop[i] = 50.0  # neutral value when undefined
    else:
        chop = np.full(n, 50.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA34_1d, ATR, volume MA, and CHOP to propagate
    start_idx = max(34, 14, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i]) or 
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema34_1d = ema_34_1d_aligned[i]
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        chop_val = chop[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        # Chop filter: only trade when market is choppy (CHOP > 61.8) or strongly trending (CHOP < 38.2)
        chop_filter = (chop_val > 61.8) or (chop_val < 38.2)
        
        if position == 0:
            # Long: price breaks above H3 AND uptrend (price > 1d EMA34) AND volume spike AND chop filter
            long_condition = (curr_close > h3) and (curr_close > ema34_1d) and volume_spike and chop_filter
            # Short: price breaks below L3 AND downtrend (price < 1d EMA34) AND volume spike AND chop filter
            short_condition = (curr_close < l3) and (curr_close < ema34_1d) and volume_spike and chop_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or price breaks below L3 (reversal signal)
            if curr_close <= entry_price - 2.0 * atr_val or curr_close < l3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or price breaks above H3 (reversal signal)
            if curr_close >= entry_price + 2.0 * atr_val or curr_close > h3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3_L3_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0
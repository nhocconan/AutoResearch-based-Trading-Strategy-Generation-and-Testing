#!/usr/bin/env python3
"""
1d Camarilla H3/L3 Breakout + 1w EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Daily Camarilla H3/L3 breakouts with volume confirmation, weekly EMA34 trend filter,
and choppiness regime filter to avoid whipsaws. Targets 7-25 trades/year to minimize fee drag.
Works in bull markets via breakout continuation and in bear markets via mean-reversion from
extreme levels when trend aligns and market is not too choppy.
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
    
    # Get 1d data for Camarilla pivots (call ONCE before loop)
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
    
    # Get 1w data for EMA34 trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate ATR(14) for stoploss
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    # Calculate 20-period volume MA for volume spike detection
    vol_ma_20 = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 19)
        vol_ma_20[i] = np.mean(volume[start_idx:i+1])
    
    # Calculate Choppiness Index (14) for regime filter
    chop = np.full(n, 50.0)  # default to neutral
    if len(close) >= 14:
        atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
        hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
        ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
        # Avoid division by zero
        range_hl = hh - ll
        range_hl = np.where(range_hl == 0, 1e-10, range_hl)
        log_term = np.log10(atr_sum / range_hl)
        chop = 100 * log_term / np.log10(14)
        chop = np.where(np.isnan(chop), 50.0, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA34_1w, ATR, volume MA, and chop to propagate
    start_idx = max(34, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema34_1w = ema_34_1w_aligned[i]
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        chop_val = chop[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        # Chop filter: avoid extreme chop (chop > 61.8) or strong trend (chop < 38.2) - we prefer ranging markets
        chop_filter = (chop_val >= 38.2) and (chop_val <= 61.8)
        
        if position == 0:
            # Long: price breaks above H3 AND uptrend (price > 1w EMA34) AND volume spike AND chop filter
            long_condition = (curr_close > h3) and (curr_close > ema34_1w) and volume_spike and chop_filter
            # Short: price breaks below L3 AND downtrend (price < 1w EMA34) AND volume spike AND chop filter
            short_condition = (curr_close < l3) and (curr_close < ema34_1w) and volume_spike and chop_filter
            
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

name = "1d_Camarilla_H3_L3_Breakout_1wEMA34_Trend_VolumeSpike_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0
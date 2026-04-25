#!/usr/bin/env python3
"""
4h Camarilla H3/L3 Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Camarilla H3/L3 breakouts with volume confirmation and 1d EMA34 trend filter.
Adds choppiness index regime filter to avoid whipsaws in sideways markets. Targets 20-40 trades/year
to minimize fee drag while capturing strong momentum moves in both bull and bear markets.
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
    
    # Get 4h data for Camarilla pivots (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Get 1d data for EMA34 trend and choppiness index (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (H3, L3) from 4h OHLC
    daily_high = df_4h['high'].values
    daily_low = df_4h['low'].values
    daily_close = df_4h['close'].values
    camarilla_h3 = daily_close + 1.1 * (daily_high - daily_low) / 4
    camarilla_l3 = daily_close - 1.1 * (daily_high - daily_low) / 4
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d choppiness index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift()).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift()).abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = tr_1d.rolling(window=14, min_periods=14).sum()
    hh_14_1d = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max()
    ll_14_1d = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min()
    chop_1d = 100 * np.log10(atr_14_1d / (hh_14_1d - ll_14_1d)) / np.log10(14)
    chop_1d = chop_1d.fillna(50).values  # neutral when undefined
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate ATR(14) for stoploss on 4h data
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
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for all indicators to propagate
    start_idx = max(34, 14)  # EMA34 needs 34, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(chop_1d_aligned[i])):
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
        chop = chop_1d_aligned[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        
        # Chop filter: only trade when market is trending (CHOP < 38.2) or extreme mean-reversion (CHOP > 61.8)
        # In trending markets (CHOP < 38.2): follow breakout direction
        # In ranging markets (CHOP > 61.8): look for reversals from extreme levels
        trending_market = chop < 38.2
        ranging_market = chop > 61.8
        
        if position == 0:
            # Long conditions
            long_breakout = (curr_close > h3) and (curr_close > ema34_1d) and volume_spike and trending_market
            long_reversal = (curr_close < l3) and (curr_close < ema34_1d) and volume_spike and ranging_market
            
            # Short conditions
            short_breakout = (curr_close < l3) and (curr_close < ema34_1d) and volume_spike and trending_market
            short_reversal = (curr_close > h3) and (curr_close > ema34_1d) and volume_spike and ranging_market
            
            if long_breakout or long_reversal:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout or short_reversal:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or reversal signal
            if curr_close <= entry_price - 2.0 * atr_val or \
               (ranging_market and curr_close < l3) or \
               (trending_market and curr_close < ema34_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or reversal signal
            if curr_close >= entry_price + 2.0 * atr_val or \
               (ranging_market and curr_close > h3) or \
               (trending_market and curr_close > ema34_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3_L3_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0
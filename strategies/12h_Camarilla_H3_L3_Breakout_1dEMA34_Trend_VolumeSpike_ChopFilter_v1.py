#!/usr/bin/env python3
"""
12h Camarilla H3/L3 Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Camarilla H3/L3 levels from 1d chart breakouts with volume confirmation,
1d EMA34 trend filter for primary trend alignment, and chop filter to avoid false breakouts.
Designed for 12h timeframe to target 12-37 trades/year. Works in bull markets via breakout
continuation and in bear markets via mean-reversion from extreme levels when trend aligns.
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
    
    # Get 1d data for Camarilla pivots and EMA34 trend (call ONCE before loop)
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
    
    # Get 1w data for choppiness regime filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        chop_regime = np.full(n, 50.0)  # Default to neutral if no weekly data
    else:
        # Calculate Choppiness Index (14-period) on weekly data
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        atr_1w = np.full(len(close_1w), np.nan)
        for i in range(1, len(close_1w)):
            tr = max(high_1w[i] - low_1w[i], 
                     abs(high_1w[i] - close_1w[i-1]),
                     abs(low_1w[i] - close_1w[i-1]))
            if i >= 1:
                atr_1w[i] = (atr_1w[i-1] * 13 + tr) / 14 if not np.isnan(atr_1w[i-1]) else tr
        
        # Choppiness Index = 100 * log10(sum(ATR))/log10(14) / log10((highest_high - lowest_low)/sum(ATR))
        chop = np.full(len(close_1w), 50.0)  # Default neutral
        for i in range(14, len(close_1w)):
            sum_atr = np.nansum(atr_1w[i-13:i+1])
            highest_high = np.nanmax(high_1w[i-13:i+1])
            lowest_low = np.nanmin(low_1w[i-13:i+1])
            if highest_high > lowest_low and sum_atr > 0:
                chop[i] = 100 * np.log10(sum_atr) / np.log10(14) / np.log10((highest_high - lowest_low) / sum_atr)
        chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
        chop_regime = chop_aligned  # We'll use raw value for filtering
    
    # Calculate ATR(14) for stoploss on 12h data
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
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA34_1d, ATR, and volume MA to propagate
    start_idx = max(34, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i])):
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
        chop_val = chop_regime[i] if len(df_1w) >= 2 else 50.0
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        
        # Choppiness filter: only trade in trending regime (CHOP < 50) for breakouts
        # In ranging markets (CHOP > 50), we avoid breakout trades to reduce false signals
        trending_regime = chop_val < 50.0
        
        if position == 0:
            # Long: price breaks above H3 AND uptrend (price > 1d EMA34) AND volume spike AND trending regime
            long_condition = (curr_close > h3) and (curr_close > ema34_1d) and volume_spike and trending_regime
            # Short: price breaks below L3 AND downtrend (price < 1d EMA34) AND volume spike AND trending regime
            short_condition = (curr_close < l3) and (curr_close < ema34_1d) and volume_spike and trending_regime
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.5*ATR below entry) or price breaks below L3 (reversal signal)
            if curr_close <= entry_price - 2.5 * atr_val or curr_close < l3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.5*ATR above entry) or price breaks above H3 (reversal signal)
            if curr_close >= entry_price + 2.5 * atr_val or curr_close > h3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3_L3_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0
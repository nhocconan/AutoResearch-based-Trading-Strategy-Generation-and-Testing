#!/usr/bin/env python3
"""
12h Camarilla H3/L3 Breakout with Volume Spike and Choppiness Filter
Hypothesis: Camarilla H3/L3 levels from 1d chart provide institutional breakout zones.
Breakouts in direction of price > 1d EMA34 with volume confirmation and low choppiness
(CHOP < 50) capture strong trending moves while avoiding whipsaws in ranging markets.
ATR-based stoploss limits drawdown. Designed for 12h timeframe targeting 15-25 trades/year.
Uses discrete position sizing (0.25) to minimize fee churn. Works in bull markets via
breakout continuation and in bear markets via mean-reversion from extreme levels when
1d trend aligns and choppiness is low. Uses proper MTF loading with get_htf_data called once before loop.
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
    
    # Get 1d data for Camarilla pivots, EMA34 trend, and choppiness filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
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
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(range)) / log10(14)
    # Low CHOP (< 38.2) = trending, High CHOP (> 61.8) = ranging
    # We use CHOP < 50 as trending regime filter
    if len(df_1d) >= 14:
        tr1 = np.abs(np.diff(daily_close, prepend=daily_close[0]))
        tr2 = np.abs(daily_high - np.roll(daily_close, 1))
        tr3 = np.abs(daily_low - np.roll(daily_close, 1))
        tr2[0] = np.abs(daily_high[0] - daily_close[0])
        tr3[0] = np.abs(daily_low[0] - daily_close[0])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr_14 = np.zeros(len(df_1d))
        for i in range(14, len(df_1d)):
            atr_14[i] = np.mean(tr[i-14:i+1])
        
        # Calculate rolling max(high) and min(low) over 14 periods
        max_high = np.zeros(len(df_1d))
        min_low = np.zeros(len(df_1d))
        for i in range(len(df_1d)):
            start_idx = max(0, i - 13)
            max_high[i] = np.max(daily_high[start_idx:i+1])
            min_low[i] = np.min(daily_low[start_idx:i+1])
        
        # Choppiness Index: 100 * log10(sum(ATR14) / (max_high - min_low)) / log10(14)
        sum_atr = np.zeros(len(df_1d))
        for i in range(len(df_1d)):
            start_idx = max(0, i - 13)
            sum_atr[i] = np.sum(atr_14[start_idx:i+1])
        
        range_hl = max_high - min_low
        # Avoid division by zero
        range_hl = np.where(range_hl == 0, 1e-10, range_hl)
        chop_raw = 100 * np.log10(sum_atr / range_hl) / np.log10(14)
        chop_raw = np.where(np.isnan(chop_raw) | np.isinf(chop_raw), 50, chop_raw)  # Default to neutral
        chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    else:
        chop_aligned = np.full(n, 50.0)  # Default to neutral if insufficient data
    
    # Calculate ATR(14) for stoploss on 12h data
    if len(close) >= 14:
        tr1 = np.abs(np.diff(close, prepend=close[0]))
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr2[0] = np.abs(high[0] - close[0])
        tr3[0] = np.abs(low[0] - close[0])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.zeros(n)
        for i in range(14, n):
            atr[i] = np.mean(tr[i-14:i+1])
        atr[:14] = np.nan
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
    
    # Start index: need enough for EMA34_1d, ATR, CHOP, and volume MA to propagate
    start_idx = max(34, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i]) or
            np.isnan(chop_aligned[i])):
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
        chop = chop_aligned[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        # Trending regime: CHOP < 50 (lower values = more trending)
        trending_regime = chop < 50
        
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
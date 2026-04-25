#!/usr/bin/env python3
"""
1h Camarilla H3/L3 Breakout + 4h EMA50 Trend + Volume Spike
Hypothesis: Camarilla H3/L3 levels from 4h chart breakouts with volume confirmation,
4h EMA50 trend filter for primary trend alignment. Uses 1h primary timeframe with 4h HTF for
trend and 1d HTF for regime filtering (choppiness index). Targets 15-37 trades/year to minimize
fee drag while capturing momentum in both bull and bear markets. Works in bull markets via
breakout continuation and in bear markets via mean-reversion from extreme levels when trend aligns.
Session filter (08-20 UTC) reduces noise trades outside active market hours.
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
    
    # Get 4h data for Camarilla pivots and EMA50 trend (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (H3, L3) from 4h OHLC
    # Camarilla: H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    daily_high = df_4h['high'].values
    daily_low = df_4h['low'].values
    daily_close = df_4h['close'].values
    camarilla_h3 = daily_close + 1.1 * (daily_high - daily_low) / 4
    camarilla_l3 = daily_close - 1.1 * (daily_high - daily_low) / 4
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for choppiness regime filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        chop_regime = np.ones(n) * 50.0  # Default to neutral regime if no daily data
    else:
        # Calculate Choppiness Index (14-period) on daily data
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        atr_1d = np.zeros(len(close_1d))
        for i in range(1, len(close_1d)):
            tr = max(high_1d[i] - low_1d[i], 
                     abs(high_1d[i] - close_1d[i-1]),
                     abs(low_1d[i] - close_1d[i-1]))
            atr_1d[i] = (atr_1d[i-1] * 13 + tr) / 14 if i >= 14 else tr
        atr_1d[:14] = np.nan
        
        # Choppiness Index = 100 * log10(sum(ATR)/log10(N)) / log10((highest_high - lowest_low) / sum(ATR))
        chop = np.full(len(close_1d), 50.0)  # Default neutral
        for i in range(14, len(close_1d)):
            sum_atr = np.nansum(atr_1d[i-13:i+1])
            highest_high = np.nanmax(high_1d[i-13:i+1])
            lowest_low = np.nanmin(low_1d[i-13:i+1])
            if highest_high > lowest_low and sum_atr > 0:
                chop[i] = 100 * np.log10(sum_atr) / np.log10(14) / np.log10((highest_high - lowest_low) / sum_atr)
        chop_regime = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate ATR(14) for stoploss on 1h data
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
    
    # Session filter: 08-20 UTC (active market hours)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA50_4h, ATR, and volume MA to propagate
    start_idx = max(50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(atr[i]) or
            np.isnan(chop_regime[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema50_4h = ema_50_4h_aligned[i]
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        chop_val = chop_regime[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        
        # Choppiness filter: only trade in trending regime (CHOP < 50) for breakouts
        # In ranging markets (CHOP > 50), we avoid breakout trades to reduce false signals
        trending_regime = chop_val < 50.0
        
        if position == 0 and in_session:
            # Long: price breaks above H3 AND uptrend (price > 4h EMA50) AND volume spike AND trending regime
            long_condition = (curr_close > h3) and (curr_close > ema50_4h) and volume_spike and trending_regime
            # Short: price breaks below L3 AND downtrend (price < 4h EMA50) AND volume spike AND trending_regime
            short_condition = (curr_close < l3) and (curr_close < ema50_4h) and volume_spike and trending_regime
            
            if long_condition:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or price breaks below L3 (reversal signal)
            if curr_close <= entry_price - 2.0 * atr_val or curr_close < l3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or price breaks above H3 (reversal signal)
            if curr_close >= entry_price + 2.0 * atr_val or curr_close > h3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3_L3_Breakout_4hEMA50_Trend_VolumeSpike_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0
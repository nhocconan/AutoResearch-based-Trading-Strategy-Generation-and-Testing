#!/usr/bin/env python3
"""
6h Camarilla R3/S3 Breakout + 1d EMA34 Trend + Volume Spike + 1w Pivot Direction Filter
Hypothesis: Camarilla R3/S3 breakouts on 6h chart with volume confirmation, aligned with 1d EMA34 trend filter, and further filtered by weekly pivot direction (bullish if price > weekly pivot, bearish if price < weekly pivot). This strategy aims to capture strong momentum moves in both bull and bear markets while avoiding false breakouts during ranging conditions. The weekly pivot filter ensures we only trade in the direction of the higher timeframe bias, improving win rate and reducing whipsaws. Targets 12-37 trades/year on 6h timeframe to minimize fee drag.
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
    
    # Get 1d data for EMA34 trend filter and weekly pivot (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for weekly pivot direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate weekly pivot point (standard: (H+L+C)/3) and R1/S1 for direction
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Get 6h data for Camarilla pivots
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (R3, S3) from 6h OHLC
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    sixh_high = df_6h['high'].values
    sixh_low = df_6h['low'].values
    sixh_close = df_6h['close'].values
    camarilla_r3 = sixh_close + 1.1 * (sixh_high - sixh_low) / 2
    camarilla_s3 = sixh_close - 1.1 * (sixh_high - sixh_low) / 2
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_s3)
    
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
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA34_1d, ATR, and volume MA to propagate
    start_idx = max(34, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema34_1d = ema_34_1d_aligned[i]
        weekly_pivot = weekly_pivot_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        
        # Weekly pivot direction: bullish if price > weekly pivot, bearish if price < weekly pivot
        weekly_bullish = curr_close > weekly_pivot
        weekly_bearish = curr_close < weekly_pivot
        
        if position == 0:
            # Long: price breaks above R3 AND uptrend (price > 1d EMA34) AND volume spike AND weekly bullish bias
            long_condition = (curr_close > r3) and (curr_close > ema34_1d) and volume_spike and weekly_bullish
            # Short: price breaks below S3 AND downtrend (price < 1d EMA34) AND volume spike AND weekly bearish bias
            short_condition = (curr_close < s3) and (curr_close < ema34_1d) and volume_spike and weekly_bearish
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or price breaks below S3 (reversal signal)
            if curr_close <= entry_price - 2.0 * atr_val or curr_close < s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or price breaks above R3 (reversal signal)
            if curr_close >= entry_price + 2.0 * atr_val or curr_close > r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike_1wPivot_v1"
timeframe = "6h"
leverage = 1.0
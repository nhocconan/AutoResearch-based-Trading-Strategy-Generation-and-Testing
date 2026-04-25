#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Donchian(20) breakouts capture strong directional moves. Filtering by 1d EMA34 trend ensures alignment with higher timeframe momentum. Volume spike confirms breakout strength. Choppiness Index (CHOP>61.8) avoids false breakouts in ranging markets. Designed for 4h timeframe targeting 20-50 trades/year. Uses discrete position sizing (0.30) to minimize fee churn. Works in bull markets via trend continuation and in bear markets via avoiding whipsaws in ranging conditions. Uses proper MTF loading with get_htf_data called once before loop.
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
    
    # Get 1d data for EMA34 trend filter and Choppiness Index (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (max(high)-min(low)))) over period
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        if len(high_arr) < period + 1:
            return np.full(len(high_arr), np.nan)
        tr1 = np.abs(np.diff(close_arr, prepend=close_arr[0]))
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr2[0] = np.abs(high_arr[0] - close_arr[0])
        tr3[0] = np.abs(low_arr[0] - close_arr[0])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.zeros(len(tr))
        for i in range(period, len(tr)):
            atr[i] = np.mean(tr[i-period+1:i+1])
        # For first period-1 values, ATR is not defined
        atr[:period-1] = np.nan
        # Calculate CHOP over the same period
        chop = np.full(len(high_arr), np.nan)
        for i in range(period-1, len(high_arr)):
            if np.isnan(atr[i]):
                chop[i] = np.nan
                continue
            sum_atr = np.nansum(atr[i-period+1:i+1])
            max_high = np.nanmax(high_arr[i-period+1:i+1])
            min_low = np.nanmin(low_arr[i-period+1:i+1])
            range_hl = max_high - min_low
            if range_hl == 0 or np.isnan(sum_atr) or np.isnan(range_hl):
                chop[i] = np.nan
            else:
                chop[i] = 100 * np.log10(sum_atr) / (np.log10(period) * range_hl)
        return chop
    
    chop_1d = calculate_chop(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate ATR(14) for stoploss on 4h data
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
    
    # Calculate Donchian(20) channels on 4h data
    def donchian_channels(high_arr, low_arr, period=20):
        upper = np.full(len(high_arr), np.nan)
        lower = np.full(len(low_arr), np.nan)
        for i in range(period-1, len(high_arr)):
            upper[i] = np.max(high_arr[i-period+1:i+1])
            lower[i] = np.min(low_arr[i-period+1:i+1])
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channels(high, low, 20)
    
    # Calculate 20-period volume MA for volume spike detection
    vol_ma_20 = np.full(n, np.nan)
    for i in range(n):
        start_idx = max(0, i - 19)
        vol_ma_20[i] = np.mean(volume[start_idx:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA34_1d, CHOP, ATR, Donchian, and volume MA to propagate
    start_idx = max(34, 14, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
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
        chop = chop_1d_aligned[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        # Chop filter: avoid ranging markets (CHOP > 61.8 = ranging)
        chop_filter = chop <= 61.8  # Only allow when not ranging (trending)
        
        if position == 0:
            # Long: price breaks above Donchian upper AND uptrend (price > 1d EMA34) AND volume spike AND not ranging
            long_condition = (curr_close > upper) and (curr_close > ema34_1d) and volume_spike and chop_filter
            # Short: price breaks below Donchian lower AND downtrend (price < 1d EMA34) AND volume spike AND not ranging
            short_condition = (curr_close < lower) and (curr_close < ema34_1d) and volume_spike and chop_filter
            
            if long_condition:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or price breaks below Donchian lower (reversal signal)
            if curr_close <= entry_price - 2.0 * atr_val or curr_close < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or price breaks above Donchian upper (reversal signal)
            if curr_close >= entry_price + 2.0 * atr_val or curr_close > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0
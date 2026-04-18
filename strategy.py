# State your hypothesis in a comment at the top (strategy type, timeframe, why it should work in BOTH bull AND bear)
# Hypothesis: 12h timeframe with 1d Donchian breakout + volume spike + ADX trend filter
# Why should work: Donchian breakouts capture momentum; volume confirms institutional interest; ADX filters whipsaws
# 12h reduces trade frequency vs 4h, minimizing fee drag while capturing multi-day moves in both bull/bear markets
# Works in bull: catches breakouts; works in bear: avoids false breaks via ADX/volume, shorts via breakdowns

#!/usr/bin/env python3
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
    
    # Get 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Donchian channels (20-period)
    donch_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (wait for 1d bar close)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Calculate 1d ADX for trend strength (14-period)
    def calculate_adx(high_arr, low_arr, close_arr, period=14):
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        up_move = high_arr - np.roll(high_arr, 1)
        down_move = np.roll(low_arr, 1) - low_arr
        up_move[0] = 0
        down_move[0] = 0
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        def smm(arr, period):
            result = np.full_like(arr, np.nan, dtype=float)
            if len(arr) >= period:
                result[period-1] = np.nansum(arr[:period])
                for i in range(period, len(arr)):
                    if not np.isnan(result[i-1]):
                        result[i] = result[i-1] - (result[i-1] / period) + arr[i]
            return result
        
        tr_smooth = smm(tr, period)
        plus_dm_smooth = smm(plus_dm, period)
        minus_dm_smooth = smm(minus_dm, period)
        
        plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
        minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
        
        dx = np.where((plus_di + minus_di) != 0, 
                      100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = smm(dx, period)
        return adx
    
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume spike detection (2x 4-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_break = donch_high_aligned[i]
        lower_break = donch_low_aligned[i]
        strong_trend = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above 1d Donchian high with volume spike and strong trend
            if (price > upper_break and volume_spike[i] and strong_trend):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian low with volume spike and strong trend
            elif (price < lower_break and volume_spike[i] and strong_trend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price crosses below 1d Donchian low (mean reversion signal)
            if price < lower_break:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price crosses above 1d Donchian high (mean reversion signal)
            if price > upper_break:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_DonchianBreakout_VolumeSpike_ADXFilter"
timeframe = "12h"
leverage = 1.0
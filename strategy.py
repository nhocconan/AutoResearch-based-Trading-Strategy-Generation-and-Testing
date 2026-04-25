#!/usr/bin/env python3
"""
6h Ichimoku Cloud Breakout + 1d ADX Trend + Volume Spike
Hypothesis: Ichimoku cloud acts as dynamic support/resistance. Breaks above/below cloud with
1d ADX > 25 (strong trend) and volume spike capture momentum in both bull/bear markets.
Cloud filter prevents whipsaws in ranging markets. ADX ensures we only trade strong trends.
Target: 12-37 trades/year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for trend strength
    # TR calculation
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift()).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    
    # +DM and -DM
    up_move = pd.Series(df_1d['high']).diff()
    down_move = pd.Series(df_1d['low']).diff().abs()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    tr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).mean().values if len(atr_14) >= 14 else np.full(len(atr_14), np.nan)
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # DI and ADX
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Current cloud boundaries (Senkou Span A/B shifted back to align with current price)
    # For breakout detection, we compare price to current cloud (Senkou A/B from 26 periods ago)
    senkou_a_current = np.roll(senkou_a, 26)
    senkou_b_current = np.roll(senkou_b, 26)
    # Fill first 26 values with NaN
    senkou_a_current[:26] = np.nan
    senkou_b_current[:26] = np.nan
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_current, senkou_b_current)
    cloud_bottom = np.minimum(senkou_a_current, senkou_b_current)
    
    # Calculate ATR(14) for stoploss
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for data to propagate
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(tenkan[i]) or 
            np.isnan(kijun[i]) or 
            np.isnan(cloud_top[i]) or 
            np.isnan(cloud_bottom[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        adx_val = adx_aligned[i]
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        cloud_top_val = cloud_top[i]
        cloud_bottom_val = cloud_bottom[i]
        atr_val = atr[i]
        
        # Volume spike: current volume > 2.0 * 30-period average
        if i >= 30:
            vol_ma_30 = np.mean(volume[i-29:i+1])
        else:
            vol_ma_30 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_30
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_val > 25
        
        # Ichimoku signals
        price_above_cloud = curr_close > cloud_top_val
        price_below_cloud = curr_close < cloud_bottom_val
        bullish_tk_cross = tenkan_val > kijun_val
        bearish_tk_cross = tenkan_val < kijun_val
        
        if position == 0:
            # Long: price breaks above cloud AND bullish TK cross AND strong trend AND volume spike
            long_condition = price_above_cloud and bullish_tk_cross and strong_trend and volume_spike
            # Short: price breaks below cloud AND bearish TK cross AND strong trend AND volume spike
            short_condition = price_below_cloud and bearish_tk_cross and strong_trend and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.5*ATR below entry) or price re-enters cloud or trend weakens
            if (curr_close <= entry_price - 2.5 * atr_val or 
                curr_close < cloud_top_val or 
                adx_val < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.5*ATR above entry) or price re-enters cloud or trend weakens
            if (curr_close >= entry_price + 2.5 * atr_val or 
                curr_close > cloud_bottom_val or 
                adx_val < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dADX_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0
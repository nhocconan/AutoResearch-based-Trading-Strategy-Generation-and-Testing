#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with TK Cross + 1d ADX Trend Filter + Volume Confirmation
# Uses Ichimoku (Tenkan/Kijun/Senkou Span A/B) for trend and momentum, filtered by 1d ADX > 25
# to ensure trading only in strong trends. Volume spike confirms institutional participation.
# Works in both bull and bear via ADX trend filter (long when +DI > -DI, short when -DI > +DI).
# Discrete sizing 0.25 to control fees. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Ichimoku_TK_Cross_1dADX_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate Ichimoku components (9, 26, 52 periods)
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
    
    # Current Ichimoku Cloud (Senkou Span A/B from 26 periods ago)
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Calculate 1d ADX for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # ADX calculation: +DI, -DI, DX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values (14-period)
    tr14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI values
    plus_di = 100 * dm_plus14 / tr14
    minus_di = 100 * dm_minus14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align HTF ADX to LTF (6h)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    plus_di_1d_aligned = align_htf_to_ltf(prices, df_1d, plus_di)
    minus_di_1d_aligned = align_htf_to_ltf(prices, df_1d, minus_di)
    
    # Volume confirmation: volume > 2.0x 24-period average (strict to reduce trades)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma_24)
    
    # ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 52, 26, 24, 14)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready or outside session
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a_shifted[i]) or
            np.isnan(senkou_b_shifted[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(plus_di_1d_aligned[i]) or np.isnan(minus_di_1d_aligned[i]) or
            np.isnan(vol_ma_24[i]) or np.isnan(atr_14[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_tenkan = tenkan[i]
        curr_kijun = kijun[i]
        curr_senkou_a = senkou_a_shifted[i]
        curr_senkou_b = senkou_b_shifted[i]
        curr_adx = adx_1d_aligned[i]
        curr_plus_di = plus_di_1d_aligned[i]
        curr_minus_di = minus_di_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_atr = atr_14[i]
        
        # Determine cloud top and bottom
        cloud_top = max(curr_senkou_a, curr_senkou_b)
        cloud_bottom = min(curr_senkou_a, curr_senkou_b)
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with ADX > 25 (strong trend) and TK cross
            if curr_volume_spike and curr_adx > 25:
                # Bullish: TK cross up (Tenkan > Kijun) + price above cloud + +DI > -DI
                if (curr_tenkan > curr_kijun and 
                    curr_close > cloud_top and 
                    curr_plus_di > curr_minus_di):
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish: TK cross down (Tenkan < Kijun) + price below cloud + -DI > +DI
                elif (curr_tenkan < curr_kijun and 
                      curr_close < cloud_bottom and 
                      curr_minus_di > curr_plus_di):
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2 * ATR below entry
            stop_loss = entry_price - 2.0 * curr_atr
            # Exit: Stoploss hit OR TK cross down OR price drops below cloud OR ADX weakens
            if (curr_low <= stop_loss or 
                curr_tenkan < curr_kijun or 
                curr_close < cloud_bottom or 
                curr_adx < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_loss = entry_price + 2.0 * curr_atr
            # Exit: Stoploss hit OR TK cross up OR price rises above cloud OR ADX weakens
            if (curr_high >= stop_loss or 
                curr_tenkan > curr_kijun or 
                curr_close > cloud_top or 
                curr_adx < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
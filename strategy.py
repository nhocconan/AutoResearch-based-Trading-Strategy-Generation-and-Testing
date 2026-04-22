#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d ADX filter and volume confirmation.
Long when Tenkan-sen crosses above Kijun-sen, price is above Kumo cloud,
ADX indicates strong trend, and volume is above average.
Short when Tenkan-sen crosses below Kijun-sen, price is below Kumo cloud,
ADX indicates strong trend, and volume is above average.
Exit when Tenkan-sen crosses back in opposite direction or price enters cloud.
Uses 1d ADX to avoid whipsaws in ranging markets, targeting 20-40 trades/year.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for ADX filter - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Kumo cloud boundaries (shifted forward by 26 periods)
    senkou_span_a_shifted = senkou_span_a.shift(26)
    senkou_span_b_shifted = senkou_span_b.shift(26)
    
    # Calculate 1d ADX (14-period)
    high_d = pd.Series(df_daily['high'].values)
    low_d = pd.Series(df_daily['low'].values)
    close_d = pd.Series(df_daily['close'].values)
    
    # True Range
    tr1 = high_d - low_d
    tr2 = abs(high_d - close_d.shift(1))
    tr3 = abs(low_d - close_d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_d = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = high_d.diff()
    down_move = -low_d.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / atr_d)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / atr_d)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_d = dx.rolling(window=14, min_periods=14).mean()
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx_d.values)
    
    # Calculate 6h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after longest lookback
        # Skip if data not ready
        if (np.isnan(tenkan_sen.iloc[i]) or np.isnan(kijun_sen.iloc[i]) or 
            np.isnan(senkou_span_a_shifted.iloc[i]) or np.isnan(senkou_span_b_shifted.iloc[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Tenkan crosses above Kijun, price above cloud, strong ADX, volume spike
            tenkan_cross_above = (tenkan_sen.iloc[i] > kijun_sen.iloc[i] and 
                                  tenkan_sen.iloc[i-1] <= kijun_sen.iloc[i-1])
            price_above_cloud = (close[i] > senkou_span_a_shifted.iloc[i] and 
                                 close[i] > senkou_span_b_shifted.iloc[i])
            if (tenkan_cross_above and price_above_cloud and 
                adx_aligned[i] > 25 and volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun, price below cloud, strong ADX, volume spike
            elif (tenkan_sen.iloc[i] < kijun_sen.iloc[i] and 
                  tenkan_sen.iloc[i-1] >= kijun_sen.iloc[i-1]):
                price_below_cloud = (close[i] < senkou_span_a_shifted.iloc[i] and 
                                     close[i] < senkou_span_b_shifted.iloc[i])
                if (price_below_cloud and adx_aligned[i] > 25 and 
                    volume[i] > 1.5 * vol_avg_20[i]):
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Tenkan crosses below Kijun OR price enters cloud
                tenkan_cross_below = (tenkan_sen.iloc[i] < kijun_sen.iloc[i] and 
                                      tenkan_sen.iloc[i-1] >= kijun_sen.iloc[i-1])
                price_in_cloud = not (close[i] > senkou_span_a_shifted.iloc[i] and 
                                      close[i] > senkou_span_b_shifted.iloc[i])
                if tenkan_cross_below or price_in_cloud or adx_aligned[i] < 20:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Tenkan crosses above Kijun OR price enters cloud
                tenkan_cross_above = (tenkan_sen.iloc[i] > kijun_sen.iloc[i] and 
                                      tenkan_sen.iloc[i-1] <= kijun_sen.iloc[i-1])
                price_in_cloud = not (close[i] < senkou_span_a_shifted.iloc[i] and 
                                      close[i] < senkou_span_b_shifted.iloc[i])
                if tenkan_cross_above or price_in_cloud or adx_aligned[i] < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_IchimokuCloud_1dADX_Volume"
timeframe = "6h"
leverage = 1.0
#%%
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation
# Ichimoku provides multi-dimensional support/resistance (cloud, conversion/base lines)
# 1d ADX filter ensures trading only in trending markets to avoid whipsaws in ranging conditions
# Volume spike (>1.5x 20-period average) confirms momentum behind breakouts
# Designed for 6h timeframe targeting 25-35 trades/year with strong performance in both bull and bear markets
# Works in bull markets via cloud breakouts, works in bear markets via ADX filter avoiding false signals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Ichimoku and ADX (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku Components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Chikou Span (Lagging Span): current close plotted 26 periods back
    # For signal generation, we use current price vs cloud
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Cloud Top and Bottom
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # 1d ADX for trend strength filter (avoid ranging markets)
    # ADX calculation: +DI, -DI, DX then smoothed
    period14_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    period14_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    up_move = period14_high[1:] - period14_high[:-1]
    down_move = period14_low[:-1] - period14_low[1:]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # True Range
    tr1 = period14_high[1:] - period14_low[1:]
    tr2 = np.abs(period14_high[1:] - close_1d[:-1])
    tr3 = np.abs(period14_low[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def WilderSmooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    if len(plus_dm) >= 14:
        plus_dm_smooth = WilderSmooth(plus_dm, 14)
        minus_dm_smooth = WilderSmooth(minus_dm, 14)
        tr_smooth = WilderSmooth(tr, 14)
        
        plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
        minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
        
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = WilderSmooth(dx, 14)
    else:
        adx = np.full_like(close_1d, np.nan)
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above cloud + bullish TK cross + strong trend (ADX > 25) + volume spike
            if (close[i] > cloud_top[i] and 
                tenkan_sen_aligned[i] > kijun_sen_aligned[i] and 
                adx_aligned[i] > 25 and 
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below cloud + bearish TK cross + strong trend (ADX > 25) + volume spike
            elif (close[i] < cloud_bottom[i] and 
                  tenkan_sen_aligned[i] < kijun_sen_aligned[i] and 
                  adx_aligned[i] > 25 and 
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to cloud or trend weakens
            if position == 1:
                # Exit long: price drops below cloud bottom or TK cross turns bearish or ADX < 20
                if (close[i] < cloud_bottom[i] or 
                    tenkan_sen_aligned[i] < kijun_sen_aligned[i] or 
                    adx_aligned[i] < 20):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price rises above cloud top or TK cross turns bullish or ADX < 20
                if (close[i] > cloud_top[i] or 
                    tenkan_sen_aligned[i] > kijun_sen_aligned[i] or 
                    adx_aligned[i] < 20):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TKCross_ADX25_VolumeConfirm"
timeframe = "6h"
leverage = 1.0
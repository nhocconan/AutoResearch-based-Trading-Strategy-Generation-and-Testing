#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 4h data for trend and structure
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h Donchian channel (20-period)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # 4h EMA(34) for trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Load 1d data for volume confirmation and regime
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    close_1d = df_1d['close'].values
    
    # 1d volume ratio (current / 20-period average)
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / np.where(vol_ma_20_1d == 0, 1, vol_ma_20_1d)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # 1d ADX(14) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    atr_smooth = pd.Series(atr_14_1d).ewm(alpha=1/14, adjust=False).mean().values
    
    plus_di = 100 * plus_dm_smooth / np.where(atr_smooth == 0, 1, atr_smooth)
    minus_di = 100 * minus_dm_smooth / np.where(atr_smooth == 0, 1, atr_smooth)
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1, (plus_di + minus_di))
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 1h price and volume
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h volume filter
    vol_ma_20_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1h = volume / np.where(vol_ma_20_1h == 0, 1, vol_ma_20_1h)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(60, n):
        # Skip if NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ratio_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        donch_high = donchian_high_aligned[i]
        donch_low = donchian_low_aligned[i]
        ema_trend = ema_34_4h_aligned[i]
        vol_ratio_1d = vol_ratio_1d_aligned[i]
        adx_val = adx_aligned[i]
        vol_ratio_1h = vol_ratio_1h[i]
        
        # Breakout conditions
        breakout_up = price > donch_high
        breakout_down = price < donch_low
        
        # Trend filter: price above/below EMA
        trend_up = price > ema_trend
        trend_down = price < ema_trend
        
        # Volume and trend strength filters
        vol_filter = vol_ratio_1h > 1.5 and vol_ratio_1d > 1.3
        trend_filter = adx_val > 25
        
        if position == 0:
            # Enter long on upward breakout with trend and volume
            if breakout_up and trend_up and vol_filter and trend_filter:
                signals[i] = 0.25
                position = 1
            # Enter short on downward breakout with trend and volume
            elif breakout_down and trend_down and vol_filter and trend_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: breakdown below Donchian low or trend reversal
            if price < donch_low or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: breakout above Donchian high or trend reversal
            if price > donch_high or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_EMA_Volume_ADX_v1"
timeframe = "4h"
leverage = 1.0
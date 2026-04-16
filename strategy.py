#!/usr/bin/env python3
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
    
    # === 4h data (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h Donchian(20) for breakout detection
    high_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper_4h = align_htf_to_ltf(prices, df_4h, high_20_4h)
    donchian_lower_4h = align_htf_to_ltf(prices, df_4h, low_20_4h)
    
    # 4h ATR for volatility filter and stoploss
    tr4h = np.maximum(high_4h - low_4h,
                      np.maximum(np.abs(high_4h - np.roll(close_4h, 1)),
                                 np.abs(low_4h - np.roll(close_4h, 1))))
    tr4h[0] = np.inf
    atr_4h = pd.Series(tr4h).rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # === 1d data (HTF for trend and regime) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d ADX for regime filter (trending vs ranging)
    # Calculate +DI and -DI
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # True Range for ADX
    tr1d = np.maximum(high_1d - low_1d,
                      np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                 np.abs(low_1d - np.roll(close_1d, 1))))
    tr1d[0] = np.inf
    
    # Smoothed values
    atr_1d_adx = pd.Series(tr1d).ewm(alpha=1/14, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / (atr_1d_adx + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / (atr_1d_adx + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === Volume confirmation (1h) ===
    df_1h = get_htf_data(prices, '1h')
    volume_1h = df_1h['volume'].values
    vol_ma_20 = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1h = volume_1h / (vol_ma_20 + 1e-10)
    vol_ratio_1h_aligned = align_htf_to_ltf(prices, df_1h, vol_ratio_1h)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_4h[i]) or np.isnan(donchian_lower_4h[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ratio_1h_aligned[i]) or np.isnan(atr_4h_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = close[i]
        upper_4h = donchian_upper_4h[i]
        lower_4h = donchian_lower_4h[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        adx_val = adx_aligned[i]
        vol_ratio_val = vol_ratio_1h_aligned[i]
        atr_4h_val = atr_4h_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below Donchian lower OR ADX weakens
            if (price < lower_4h) or (adx_val < 20):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above Donchian upper OR ADX weakens
            if (price > upper_4h) or (adx_val < 20):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Only trade during session
            if in_session:
                # LONG: Price breaks above Donchian upper AND above EMA50 (trend filter)
                # AND strong trend (ADX > 25) AND volume spike
                if (price > upper_4h) and (price > ema_50_1d_val) and \
                   (adx_val > 25) and (vol_ratio_val > 1.5):
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                
                # SHORT: Price breaks below Donchian lower AND below EMA50 (trend filter)
                # AND strong trend (ADX > 25) AND volume spike
                elif (price < lower_4h) and (price < ema_50_1d_val) and \
                     (adx_val > 25) and (vol_ratio_val > 1.5):
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_Breakout_ADX_Volume"
timeframe = "4h"
leverage = 1.0
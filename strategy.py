#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h Supertrend (ATR=10, mult=3.0) for trend direction
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    prev_close_4h = np.concatenate([[close_4h[0]], close_4h[:-1]])
    tr_4h = np.maximum(high_4h - low_4h,
                       np.maximum(np.abs(high_4h - prev_close_4h),
                                  np.abs(low_4h - prev_close_4h)))
    atr_4h = pd.Series(tr_4h).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    hl2_4h = (high_4h + low_4h) / 2
    upper_basic_4h = hl2_4h + 3.0 * atr_4h
    lower_basic_4h = hl2_4h - 3.0 * atr_4h
    
    # Final Supertrend calculation
    upper_band_4h = np.full_like(close_4h, np.nan)
    lower_band_4h = np.full_like(close_4h, np.nan)
    supertrend_4h = np.full_like(close_4h, np.nan)
    trend_4h = np.full_like(close_4h, np.nan)  # 1 for uptrend, -1 for downtrend
    
    upper_band_4h[0] = upper_basic_4h[0]
    lower_band_4h[0] = lower_basic_4h[0]
    supertrend_4h[0] = upper_basic_4h[0]
    trend_4h[0] = 1
    
    for i in range(1, len(close_4h)):
        # Upper band
        if upper_basic_4h[i] < upper_band_4h[i-1] or close_4h[i-1] > upper_band_4h[i-1]:
            upper_band_4h[i] = upper_basic_4h[i]
        else:
            upper_band_4h[i] = upper_band_4h[i-1]
            
        # Lower band
        if lower_basic_4h[i] > lower_band_4h[i-1] or close_4h[i-1] < lower_band_4h[i-1]:
            lower_band_4h[i] = lower_basic_4h[i]
        else:
            lower_band_4h[i] = lower_band_4h[i-1]
            
        # Supertrend and trend
        if supertrend_4h[i-1] == upper_band_4h[i-1]:
            if close_4h[i] <= upper_band_4h[i]:
                supertrend_4h[i] = upper_band_4h[i]
                trend_4h[i] = -1
            else:
                supertrend_4h[i] = lower_band_4h[i]
                trend_4h[i] = 1
        else:
            if close_4h[i] >= lower_band_4h[i]:
                supertrend_4h[i] = lower_band_4h[i]
                trend_4h[i] = 1
            else:
                supertrend_4h[i] = upper_band_4h[i]
                trend_4h[i] = -1
    
    # Align Supertrend to 1h timeframe
    supertrend_4h_aligned = align_htf_to_ltf(prices, df_4h, supertrend_4h)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Get 1d HTF data once before loop for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for trend strength regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    prev_close_1d = np.concatenate([[close_1d[0]], close_1d[:-1]])
    tr_1d = np.maximum(high_1d - low_1d,
                       np.maximum(np.abs(high_1d - prev_close_1d),
                                  np.abs(low_1d - prev_close_1d)))
    
    # Directional Movement
    up_move_1d = high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])
    down_move_1d = np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d
    
    plus_dm_1d = np.where((up_move_1d > down_move_1d) & (up_move_1d > 0), up_move_1d, 0)
    minus_dm_1d = np.where((down_move_1d > up_move_1d) & (down_move_1d > 0), down_move_1d, 0)
    
    # Smoothed values
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm_1d).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr_1d + 1e-10)
    minus_di_1d = 100 * pd.Series(minus_dm_1d).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr_1d + 1e-10)
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
    adx_1d = pd.Series(dx_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 1h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 1h RSI(14) for entry timing
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1h volume ratio for confirmation
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_4h_aligned[i]) or np.isnan(trend_4h_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        # Entry conditions:
        # 1. 4h Supertrend trend direction (HTF signal direction)
        # 2. 1d ADX > 25 for trending regime (regime filter)
        # 3. 1h RSI pullback to opposite extreme for entry timing (lower TF timing)
        # 4. Volume confirmation > 1.2
        # 5. Session filter (08-20 UTC)
        # 6. Discrete position sizing: 0.20
        
        # Long conditions: 4h uptrend + ADX trending + RSI oversold pullback
        if (trend_4h_aligned[i] == 1 and  # 4h uptrend
            adx_1d_aligned[i] > 25 and    # Trending regime on 1d
            rsi[i] < 30 and               # Oversold on 1h for entry timing
            volume_ratio[i] > 1.2 and     # Volume confirmation
            in_session):                  # Session filter
            signals[i] = 0.20
            
        # Short conditions: 4h downtrend + ADX trending + RSI overbought pullback
        elif (trend_4h_aligned[i] == -1 and  # 4h downtrend
              adx_1d_aligned[i] > 25 and     # Trending regime on 1d
              rsi[i] > 70 and                # Overbought on 1h for entry timing
              volume_ratio[i] > 1.2 and      # Volume confirmation
              in_session):                   # Session filter
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_4hSupertrend_1dADX_RSI_Pullback_Volume_Session"
timeframe = "1h"
leverage = 1.0
# 12h_1w_volatility_breakout_v1
# Hypothesis: 12-hour volatility breakout using 1-week ATR-based channels with volume confirmation and ADX trend filter.
# Designed for low trade frequency (~15-25 trades/year) to minimize fee decay and work in both bull and bear markets.
# Uses weekly ATR for dynamic breakout levels, daily volume confirmation, and weekly ADX to filter trend strength.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_volatility_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 1-week ATR for volatility-based channels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range for weekly
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1w[0] = tr1[0]
    atr_period = 6
    atr_1w = pd.Series(tr_1w).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Dynamic channels: ATR multiplier based on volatility regime (70th percentile)
    atr_median = np.nanmedian(atr_1w)
    atr_mult = np.where(atr_1w > atr_median, 1.8, 1.2)  # Higher mult in high vol
    
    # Upper and lower bands (previous week's close ± ATR*mult)
    upper_band = np.roll(close_1w, 1) + atr_1w * atr_mult
    lower_band = np.roll(close_1w, 1) - atr_1w * atr_mult
    
    # Align bands to 12h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1w, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1w, lower_band)
    
    # Load daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily volume confirmation: current volume > 20-day average
    volume_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    # Calculate weekly ADX for trend strength filter
    # +DM and -DM
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr_1w_smooth = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI values
    plus_di = 100 * plus_dm_smooth / atr_1w_smooth
    minus_di = 100 * minus_dm_smooth / atr_1w_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 30 to ensure sufficient data
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(vol_avg_20_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current daily volume (aligned)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        vol_confirm = vol_1d_current > vol_avg_20_aligned[i]
        
        # Trend filter: only trade when ADX > 25 (trending market)
        trend_filter = adx_aligned[i] > 25
        
        price = close[i]
        upper = upper_band_aligned[i]
        lower = lower_band_aligned[i]
        
        # Entry conditions: Breakout of weekly bands with volume and trend confirmation
        long_signal = vol_confirm and trend_filter and (price > upper)
        short_signal = vol_confirm and trend_filter and (price < lower)
        
        # Exit conditions: Return to middle (previous week's close) or opposite extreme
        prev_close_aligned = align_htf_to_ltf(prices, df_1w, np.roll(close_1w, 1))[i]
        # Additional exit: if trend weakens (ADX < 20)
        trend_weak = adx_aligned[i] < 20
        long_exit = price < prev_close_aligned or trend_weak
        short_exit = price > prev_close_aligned or trend_weak
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
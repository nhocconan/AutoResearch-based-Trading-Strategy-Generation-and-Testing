#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w ADX filter and volume confirmation
# - Long when price breaks above 20-period Donchian high on 1d + ADX > 25 (trending) + volume > 1.5x average
# - Short when price breaks below 20-period Donchian low on 1d + ADX > 25 (trending) + volume > 1.5x average
# - Exit when price crosses the 10-period EMA on 1d or ADX drops below 20 (trend weakening)
# - Designed to capture strong trends in both bull and bear markets while avoiding chop
# - Target: 15-25 trades/year to minimize fee drag

name = "1d_Donchian20_ADX_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian, EMA, ADX calculations
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # 1d 10-period EMA for exit
    ema_10_1d = pd.Series(df_1d['close'].values).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # 1d ADX for trend strength (14-period)
    # Calculate True Range
    tr1 = pd.Series(high_1d).subtract(pd.Series(low_1d)).abs()
    tr2 = pd.Series(high_1d).subtract(pd.Series(df_1d['close'].shift(1))).abs()
    tr3 = pd.Series(low_1d).subtract(pd.Series(df_1d['close'].shift(1))).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    # Calculate Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = pd.Series(low_1d).diff().multiply(-1)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth the DM values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean()
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean()
    
    # Calculate DI values
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # Calculate DX and ADX
    dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], 0)
    adx = dx.rolling(window=14, min_periods=14).mean().values
    
    # Get 1w data for trend filter (additional confirmation)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all 1d indicators to 1d timeframe (no additional delay needed as we use same timeframe)
    donchian_high_aligned = donchian_high
    donchian_low_aligned = donchian_low
    ema_10_1d_aligned = ema_10_1d
    adx_aligned = adx
    
    # Align 1w EMA to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_10_1d_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter
        volume_filter = vol_ma_20[i] > 0 and volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Look for long entry: price breaks above Donchian high + strong trend (ADX > 25) + volume + price above weekly EMA50
            if (close[i] > donchian_high_aligned[i] and 
                adx_aligned[i] > 25 and 
                volume_filter and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Look for short entry: price breaks below Donchian low + strong trend (ADX > 25) + volume + price below weekly EMA50
            elif (close[i] < donchian_low_aligned[i] and 
                  adx_aligned[i] > 25 and 
                  volume_filter and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price crosses below 10 EMA OR ADX weakens (< 20) OR price breaks Donchian low
            if (close[i] < ema_10_1d_aligned[i] or 
                adx_aligned[i] < 20 or 
                close[i] < donchian_low_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price crosses above 10 EMA OR ADX weakens (< 20) OR price breaks Donchian high
            if (close[i] > ema_10_1d_aligned[i] or 
                adx_aligned[i] < 20 or 
                close[i] > donchian_high_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
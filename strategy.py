#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for calculations (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ATR (14-period) for volatility
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.roll(df_1d['close'], 1))
    tr3 = np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ADX (14-period) for trend strength
    up_move = df_1d['high'] - np.roll(df_1d['high'], 1)
    down_move = np.roll(df_1d['low'], 1) - df_1d['low']
    up_move[0] = np.nan
    down_move[0] = np.nan
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr_adx = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_adx = pd.Series(tr_adx).rolling(window=14, min_periods=14).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_adx
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_adx
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d Bollinger Bands (20, 2.0)
    sma_20_1d = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + 2 * std_20_1d
    lower_bb_1d = sma_20_1d - 2 * std_20_1d
    
    # Calculate 1d Donchian Channels (20-period)
    donch_high_1d = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donch_low_1d = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Align indicators to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    upper_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_1d)
    lower_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_1d)
    donch_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    
    # Calculate 4h Bollinger Bands (20, 2.0)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 150  # Wait for enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_1d_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(upper_bb_1d_aligned[i]) or
            np.isnan(lower_bb_1d_aligned[i]) or
            np.isnan(donch_high_1d_aligned[i]) or
            np.isnan(donch_low_1d_aligned[i]) or
            np.isnan(sma_20[i]) or
            np.isnan(std_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR > 20-period average
        if i >= 20:
            atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
            atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
            vol_filter = not np.isnan(atr_ma_1d_aligned[i]) and atr_1d_aligned[i] > atr_ma_1d_aligned[i]
        else:
            vol_filter = False
        
        # Trend filter: ADX > 25 for trending market
        trend_filter = adx_aligned[i] > 25
        
        trade_allowed = vol_filter and trend_filter
        
        if position == 0:
            # Long: Price touches lower BB in uptrend
            if trade_allowed and close[i] <= lower_band[i] and close[i] > donch_low_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price touches upper BB in downtrend
            elif trade_allowed and close[i] >= upper_band[i] and close[i] < donch_high_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price touches middle band or breaks above upper BB
            if close[i] >= sma_20[i] or close[i] >= upper_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches middle band or breaks below lower BB
            if close[i] <= sma_20[i] or close[i] <= lower_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_BB_Donchian_ADX_Filter_v1"
timeframe = "4h"
leverage = 1.0
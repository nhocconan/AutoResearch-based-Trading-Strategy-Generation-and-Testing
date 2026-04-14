#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian breakout with 1-day trend filter and volume confirmation
# Uses Donchian(20) breakout on 4h timeframe for entry
# Daily ADX(14) > 25 to filter for trending markets on 1d
# Volume > 1.5x 20-period EMA for confirmation (higher threshold to reduce trades)
# Designed for 20-50 trades/year with clear trend-following logic
# Position size: 0.25 to balance return and drawdown
# Works in bull markets via trend continuation and in bear markets via short signals

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data once
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Load daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian upper/lower bands
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Daily ADX for trend filter (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and DM
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]
    
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(high_1d)
    for i in range(1, len(high_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr_1d + 1e-10)
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr_1d + 1e-10)
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
    adx_1d = pd.Series(dx_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume moving average for confirmation (20-period EMA)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(20, n):  # Start after Donchian period
        # Get aligned 4h Donchian levels
        donch_high_i = align_htf_to_ltf(prices, df_4h, donch_high)[i]
        donch_low_i = align_htf_to_ltf(prices, df_4h, donch_low)[i]
        
        # Get aligned daily ADX
        adx_1d_i = align_htf_to_ltf(prices, df_1d, adx_1d)[i]
        
        if np.isnan(donch_high_i) or np.isnan(donch_low_i) or np.isnan(adx_1d_i) or np.isnan(vol_ma[i]):
            continue
        
        # Volume confirmation (1.5x average - higher threshold to reduce trades)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Long: Price breaks above Donchian high + daily trend up + volume
        if position == 0 and close[i] > donch_high_i and adx_1d_i > 25 and volume_confirm:
            position = 1
            signals[i] = position_size
        # Short: Price breaks below Donchian low + daily trend down + volume
        elif position == 0 and close[i] < donch_low_i and adx_1d_i > 25 and volume_confirm:
            position = -1
            signals[i] = -position_size
        # Exit: Price returns to opposite Donchian band or trend weakens
        elif position != 0:
            if position == 1 and (close[i] < donch_low_i or adx_1d_i < 20):
                position = 0
                signals[i] = 0.0
            elif position == -1 and (close[i] > donch_high_i or adx_1d_i < 20):
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian_DailyADX_Volume"
timeframe = "4h"
leverage = 1.0
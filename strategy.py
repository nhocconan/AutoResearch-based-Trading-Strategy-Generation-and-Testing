#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with weekly trend filter and volume confirmation
# Uses Donchian(20) breakout on daily timeframe for entry
# Weekly ADX(14) > 25 to filter for trending markets
# Volume > 1.3x 20-period EMA for confirmation
# Designed for 15-25 trades/year with clear trend-following logic
# Position size: 0.25 to balance return and drawdown

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian upper/lower bands
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Weekly ADX for trend filter (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range and DM
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1w[0] = tr1[0]
    
    plus_dm = np.zeros_like(high_1w)
    minus_dm = np.zeros_like(high_1w)
    for i in range(1, len(high_1w)):
        up_move = high_1w[i] - high_1w[i-1]
        down_move = low_1w[i-1] - low_1w[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    atr_1w = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_1w = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr_1w + 1e-10)
    minus_di_1w = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr_1w + 1e-10)
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w + 1e-10)
    adx_1w = pd.Series(dx_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume moving average for confirmation (20-period EMA)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(20, n):  # Start after Donchian period
        # Get aligned 1d Donchian levels
        donch_high_i = align_htf_to_ltf(prices, df_1d, donch_high)[i]
        donch_low_i = align_htf_to_ltf(prices, df_1d, donch_low)[i]
        
        # Get aligned weekly ADX
        adx_1w_i = align_htf_to_ltf(prices, df_1w, adx_1w)[i]
        
        if np.isnan(donch_high_i) or np.isnan(donch_low_i) or np.isnan(adx_1w_i) or np.isnan(vol_ma[i]):
            continue
        
        # Volume confirmation (1.3x average)
        volume_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Long: Price breaks above Donchian high + weekly trend up + volume
        if position == 0 and close[i] > donch_high_i and adx_1w_i > 25 and volume_confirm:
            position = 1
            signals[i] = position_size
        # Short: Price breaks below Donchian low + weekly trend down + volume
        elif position == 0 and close[i] < donch_low_i and adx_1w_i > 25 and volume_confirm:
            position = -1
            signals[i] = -position_size
        # Exit: Price returns to opposite Donchian band or trend weakens
        elif position != 0:
            if position == 1 and (close[i] < donch_low_i or adx_1w_i < 20):
                position = 0
                signals[i] = 0.0
            elif position == -1 and (close[i] > donch_high_i or adx_1w_i < 20):
                position = 0
                signals[i] = 0.0
    
    return signals

name = "1d_Donchian_WeeklyADX_Volume"
timeframe = "1d"
leverage = 1.0
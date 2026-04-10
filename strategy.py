#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and ADX trend filter
# - Primary: 4h price breaks above/below Donchian(20) channels (structure-based breakout)
# - Volume filter: 1d volume > 1.5x 20-period volume MA to confirm institutional participation
# - Trend filter: ADX(14) > 25 to ensure we trade only in trending markets (avoid chop)
# - Exit: Price returns to Donchian(20) midpoint (mean reversion within the channel)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Donchian adapts to volatility, volume confirms breakout quality,
#   ADX filter avoids ranging markets, effective in both bull and bear regimes

name = "4h_1d_donchian_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Donchian channels for 4h timeframe (20-period)
    # Upper channel = highest high over 20 periods
    # Lower channel = lowest low over 20 periods
    # Middle channel = (upper + lower) / 2
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Calculate 1d volume confirmation: volume > 1.5x 20-period volume MA
    volume_ma_20_1d = pd.Series(volume_1d).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate ADX(14) for trend filter
    # ADX calculation requires +DI and -DI
    # +DI = 100 * EWMASM((+DM / ATR), 14)
    # -DI = 100 * EWMASM((-DM / ATR), 14)
    # DX = 100 * ABS((+DI - -DI) / (+DI + -DI))
    # ADX = EWMASM(DX, 14)
    
    # Calculate True Range (TR)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate +DM and -DM
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Calculate smoothed values
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Calculate +DI and -DI
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)  # Avoid division by zero
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(volume_ma_20_1d_aligned[i]) or 
            np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.5x 20-period MA
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_confirm = vol_1d_current[i] > 1.5 * volume_ma_20_1d_aligned[i]
        
        # Trend filter: ADX > 25 (trending market)
        trend_filter = adx[i] > 25
        
        if position == 0:  # Flat - look for new Donchian breakouts
            # Long entry: Price breaks above upper channel + vol confirmation + trend filter
            if close[i] > donchian_upper[i] and vol_confirm and trend_filter:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below lower channel + vol confirmation + trend filter
            elif close[i] < donchian_lower[i] and vol_confirm and trend_filter:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price returns to Donchian middle (mean reversion within channel)
            if position == 1:  # Long position
                if close[i] <= donchian_middle[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] >= donchian_middle[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and ADX trend filter
# Uses Donchian(20) breakout for directional entries, confirmed by 1d volume spike (>1.5x 20-period average)
# and ADX(14) > 25 to ensure trending markets. Exits when price crosses Donchian midpoint.
# Designed for moderate trade frequency (target: 20-50 trades/year) to balance signal quality and fee drag.
# Works in bull markets via breakout continuation and in bear markets via breakdown continuation.

name = "4h_donchian20_1d_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate ADX(14) for trend strength
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Daily volume average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-day average
        volume_confirm = vol_1d_aligned[i] > 1.5 * vol_ma_1d[-1] if len(vol_ma_1d) > 0 else False
        
        # Trend filter: ADX > 25 indicates strong trend
        trending = adx[i] > 25
        
        # Long: price breaks above Donchian high with volume and trend confirmation
        if close[i] > donchian_high[i] and volume_confirm and trending:
            signals[i] = 0.30
        # Short: price breaks below Donchian low with volume and trend confirmation
        elif close[i] < donchian_low[i] and volume_confirm and trending:
            signals[i] = -0.30
        # Exit: price crosses Donchian midpoint (reversal signal)
        elif (close[i-1] > donchian_mid[i-1] and close[i] < donchian_mid[i]) or \
             (close[i-1] < donchian_mid[i-1] and close[i] > donchian_mid[i]):
            signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals
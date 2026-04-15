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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for volatility regime filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate daily ATR(50) for long-term volatility regime
    atr_50_1d = pd.Series(tr_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr_50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_50_1d)
    
    # Volatility regime: only trade when short-term ATR is between 0.5x and 2.0x long-term ATR
    # This avoids both extremely low volatility (chop) and extremely high volatility (panic)
    vol_ratio = atr_14_1d_aligned / (atr_50_1d_aligned + 1e-10)
    vol_regime = (vol_ratio >= 0.5) & (vol_ratio <= 2.0)
    
    # Calculate 4h Donchian channels (20-period) for breakout signals
    # Use rolling window on 4h data directly
    donchian_high = pd.Series(close).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(close).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h ADX(14) for trend strength filter
    # +DM and -DM
    up_move = np.diff(high, prepend=high[0])
    down_move = np.diff(low, prepend=low[0]) * -1  # invert to positive
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # True Range
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # first period
    
    # Smoothed values
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI values
    plus_di = 100 * plus_dm_smooth / (atr_14 + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Trade only when volatility regime is normal and trend is strong enough
    for i in range(100, n):
        if not vol_regime[i] or np.isnan(adx[i]) or adx[i] < 25:
            signals[i] = 0.0
            continue
            
        # Long breakout: price breaks above Donchian high with volume confirmation
        if (close[i] > donchian_high[i-1] and  # break above previous high
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i])):  # volume spike
            signals[i] = 0.25
            
        # Short breakout: price breaks below Donchian low with volume confirmation
        elif (close[i] < donchian_low[i-1] and  # break below previous low
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i])):  # volume spike
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_Breakout_Volume_ADX_v1"
timeframe = "4h"
leverage = 1.0
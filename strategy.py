#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian channel breakout with 1-day volume confirmation and ADX trend filter
# Long when price breaks above Donchian(20) high + 1-day volume > 1.5x 20-day average + ADX(14) > 25
# Short when price breaks below Donchian(20) low + 1-day volume > 1.5x 20-day average + ADX(14) > 25
# Exit when price crosses Donchian midline or ADX < 20 (trend weakening)
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 1-day volume for confirmation and 12h ADX for trend strength
# Target: 75-150 total trades over 4 years (19-38/year)

name = "12h_donchian20_1d_vol_adx_v2"
timeframe = "12h"
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
    
    # 1-day data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day average volume (20-day)
    vol_1d = df_1d['volume'].values
    vol_1d_s = pd.Series(vol_1d)
    vol_avg_20 = vol_1d_s.rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    # 12-hour Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 12-hour ADX(14) for trend strength
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_s = pd.Series(tr)
    plus_dm_s = pd.Series(plus_dm)
    minus_dm_s = pd.Series(minus_dm)
    
    atr = tr_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * (plus_dm_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / (atr + 1e-10))
    minus_di = 100 * (minus_dm_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / (atr + 1e-10))
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    dx_s = pd.Series(dx)
    adx = dx_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_avg_20_aligned[i]) or np.isnan(adx[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses midline or ADX < 20 (trend weakening)
            elif close[i] < donchian_mid[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses midline or ADX < 20 (trend weakening)
            elif close[i] > donchian_mid[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with volume confirmation and ADX trend filter
            volume_confirm = volume[i] > 1.5 * vol_avg_20_aligned[i]
            strong_trend = adx[i] > 25
            
            # Long: price breaks above Donchian high + volume confirmation + strong trend
            if close[i] > donchian_high[i] and volume_confirm and strong_trend:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low + volume confirmation + strong trend
            elif close[i] < donchian_low[i] and volume_confirm and strong_trend:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals
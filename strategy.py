#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d ADX regime filter + volume confirmation
# - Uses 4h Donchian channel breakout for trend following entries
# - Uses 1d ADX(20) to filter regime: only trade when ADX > 20 (trending market)
# - Long when price breaks above Donchian upper channel AND ADX > 20
# - Short when price breaks below Donchian lower channel AND ADX > 20
# - Volume confirmation: require current volume > 1.5x 20-period average volume
# - ATR-based stoploss: exit when price moves 2.5x ATR against position
# - Fixed position size 0.25 to control drawdown
# - Works in both bull and bear markets by only trading in trending regimes (ADX > 20)
# - Donchian breakouts capture strong moves; ADX filter avoids choppy markets
# - Volume confirmation ensures breakouts have conviction
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)

name = "4h_1d_donchian_adx_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(20)
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]  # First bar
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    tr_ma = pd.Series(tr_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    plus_dm_ma = pd.Series(plus_dm).ewm(span=20, adjust=False, min_periods=20).mean().values
    minus_dm_ma = pd.Series(minus_dm).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_ma / tr_ma
    minus_di = 100 * minus_dm_ma / tr_ma
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 4h Donchian channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h ATR (20-period) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]) or
            atr[i] <= 0 or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when ADX > 20 (trending market)
        trending_regime = adx_1d_aligned[i] > 20
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # ATR-based trailing stop: exit if price drops 2.5x ATR from highest high
            if close[i] < highest_high_since_entry - 2.5 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # ATR-based trailing stop: exit if price rises 2.5x ATR from lowest low
            if close[i] > lowest_low_since_entry + 2.5 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if trending_regime and volume_confirm:
                # Long entry: price breaks above Donchian upper channel
                if close[i] > highest_high[i]:
                    position = 1
                    highest_high_since_entry = high[i]
                    lowest_low_since_entry = low[i]
                    signals[i] = 0.25
                # Short entry: price breaks below Donchian lower channel
                elif close[i] < lowest_low[i]:
                    position = -1
                    highest_high_since_entry = high[i]
                    lowest_low_since_entry = low[i]
                    signals[i] = -0.25
    
    return signals
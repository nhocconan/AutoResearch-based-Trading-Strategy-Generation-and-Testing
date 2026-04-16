# 12h Donchian Breakout with Volume Confirmation and 1d Trend Filter
# Strategy Type: Trend-following breakout with risk management
# Timeframe: 12h (primary), 1d (trend filter)
# Hypothesis: Breakouts from 20-period Donchian channels on 12h timeframe,
# confirmed by volume expansion and aligned with 1d trend (ADX > 25),
# capture sustained moves in both bull and bear markets.
# Uses discrete position sizing (0.25) to limit trade frequency and fee drag.
# Target: 20-60 trades per symbol over 4 years (5-15/year) for low turnover.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h data (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # === 1d data (trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 12h Donchian Channel (20 periods) ===
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # === 12h Volume confirmation ===
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = volume_12h / vol_ma_20_12h
    
    # === 1d ADX(14) for trend filter ===
    # True Range
    tr1 = pd.Series(high_1d).diff()
    tr2 = abs(pd.Series(high_1d).diff())
    tr3 = abs(pd.Series(low_1d).diff())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = -pd.Series(low_1d).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth DM and TR
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ratio_12h[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        vol_ratio = vol_ratio_12h[i]
        adx_val = adx_aligned[i]
        
        # === STOPLOSS LOGIC (ATR-based) ===
        if position != 0:
            atr_12h = np.abs(high_12h - low_12h)
            atr_ma = pd.Series(atr_12h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_12h, atr_ma)
            atr_val = atr_aligned[i]
            
            if position == 1:  # Long position
                if price < entry_price - 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                    continue
            elif position == -1:  # Short position
                if price > entry_price + 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                    continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price breaks below Donchian lower channel
            if price < donch_low[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        elif position == -1:  # Short position
            # Exit when price breaks above Donchian upper channel
            if price > donch_high[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require trending market and volume confirmation
            if adx_val > 25 and vol_ratio > 1.5:
                # Long breakout: price closes above upper Donchian channel
                if close[i] > donch_high[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Short breakdown: price closes below lower Donchian channel
                elif close[i] < donch_low[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian_Breakout_Volume_ADXFilter_v1"
timeframe = "12h"
leverage = 1.0
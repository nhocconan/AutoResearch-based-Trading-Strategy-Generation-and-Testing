#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ADX(14) trend filter and volume confirmation
# Uses discrete sizing 0.25 to limit fee drag. Target: 75-200 total trades over 4 years (19-50/year).
# Donchian channels provide objective structure; breakouts with 1d ADX>25 filter ensure trending markets.
# Volume confirmation ensures institutional participation. Works in both bull and bear markets via ADX trend filter.
# Focus on BTC/ETH with proven edge from Donchian + ADX + volume confluence.

name = "4h_Donchian20_1dADX14_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian(20) from prior 20 bars
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 1d ADX(14) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift(1)).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift(1)).abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = pd.Series(df_1d['high']).diff()
    down_move = -pd.Series(df_1d['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm)
    minus_dm = pd.Series(minus_dm)
    
    # Smoothed DM and TR
    plus_di_1d = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_1d)
    minus_di_1d = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_1d)
    dx_1d = 100 * (abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d))
    adx_1d = dx_1d.ewm(alpha=1/14, adjust=False).mean()
    adx_1d_values = adx_1d.values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_values)
    
    # Volume confirmation: volume > 2.0x 20-period average (strict to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 20, 14, 20, 14)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_adx_1d = adx_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_atr = atr_14[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with Donchian break and 1d ADX>25 trend filter
            if curr_volume_spike and curr_adx_1d > 25:
                # Bullish: Close breaks above Donchian high
                if curr_close > curr_donchian_high:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish: Close breaks below Donchian low
                elif curr_close < curr_donchian_low:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2 * ATR below entry
            stop_loss = entry_price - 2.0 * curr_atr
            # Exit: Stoploss hit OR close drops below Donchian low OR ADX < 20 (trend weak)
            if curr_low <= stop_loss or curr_close < curr_donchian_low or curr_adx_1d < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_loss = entry_price + 2.0 * curr_atr
            # Exit: Stoploss hit OR close rises above Donchian high OR ADX < 20 (trend weak)
            if curr_high >= stop_loss or curr_close > curr_donchian_high or curr_adx_1d < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
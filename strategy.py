#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d volume spike and ADX regime filter
# - Entry: Long when price breaks above Donchian upper (20) + 1d volume > 1.5x 20-period average + 1d ADX > 25
#          Short when price breaks below Donchian lower (20) + 1d volume > 1.5x 20-period average + 1d ADX > 25
# - Exit: Close-based reversal - exit long when price < Donchian lower (20), exit short when price > Donchian upper (20)
# - Stoploss: ATR-based - exit when price moves against position by 2.0 * ATR(14) on 4h
# - Position sizing: 0.25 (discrete level)
# - Target: 75-200 total trades over 4 years (19-50/year) to stay within HARD MAX: 400 total
# - Donchian channels provide clear trend structure, volume confirmation ensures participation,
#   ADX>25 filters for strong trending markets (reduces whipsaw in chop)
# - Works in bull markets via upside breakouts and in bear markets via downside breakdowns

name = "4h_1d_donchian_breakout_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Pre-compute 1d data for Donchian, volume and ADX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ADX (14-period)
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM
    alpha = 1.0 / 14
    atr_1d = np.zeros_like(tr)
    plus_dm_1d = np.zeros_like(plus_dm)
    minus_dm_1d = np.zeros_like(minus_dm)
    
    atr_1d[0] = tr[0]
    plus_dm_1d[0] = plus_dm[0]
    minus_dm_1d[0] = minus_dm[0]
    
    for i in range(1, len(tr)):
        atr_1d[i] = (1 - alpha) * atr_1d[i-1] + alpha * tr[i]
        plus_dm_1d[i] = (1 - alpha) * plus_dm_1d[i-1] + alpha * plus_dm[i]
        minus_dm_1d[i] = (1 - alpha) * minus_dm_1d[i-1] + alpha * minus_dm[i]
    
    # Avoid division by zero
    plus_di_1d = np.where(atr_1d > 0, 100 * plus_dm_1d / atr_1d, 0.0)
    minus_di_1d = np.where(atr_1d > 0, 100 * minus_dm_1d / atr_1d, 0.0)
    
    dx_1d = np.where((plus_di_1d + minus_di_1d) > 0, 
                     100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d), 
                     0.0)
    
    # Smoothed DX to get ADX
    adx_1d = np.zeros_like(dx_1d)
    if len(dx_1d) >= 28:
        adx_1d[13] = np.mean(dx_1d[14:28])
    elif len(dx_1d) > 14:
        adx_1d[13] = np.mean(dx_1d[14:])
    else:
        adx_1d[13] = 0.0
    for i in range(14, len(dx_1d)):
        adx_1d[i] = (1 - alpha) * adx_1d[i-1] + alpha * dx_1d[i]
    
    # Calculate 4h ATR (14-period) for stoploss
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr1_4h[0] = 0
    tr2_4h[0] = 0
    tr3_4h[0] = 0
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    # Align all HTF data to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    atr_4h_aligned = align_htf_to_ltf(prices, prices, atr_4h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(atr_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 4h close
        close_price = close_4h[i]
        
        # Get current 1d volume for confirmation
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirmation = volume_1d_aligned[i] > 1.5 * volume_ma_aligned[i]
        
        # ADX filter: > 25 indicates strong trend strength (reduces whipsaw)
        adx_filter = adx_aligned[i] > 25.0
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper + volume confirmation + ADX filter
            if (close_price > donchian_high_aligned[i] and 
                volume_confirmation and 
                adx_filter):
                position = 1
                entry_price = close_price
                signals[i] = 0.25
            # Short entry: price breaks below Donchian lower + volume confirmation + ADX filter
            elif (close_price < donchian_low_aligned[i] and 
                  volume_confirmation and 
                  adx_filter):
                position = -1
                entry_price = close_price
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or stoploss
            # Calculate stoploss level
            if position == 1:  # Long position
                stop_loss = entry_price - 2.0 * atr_4h_aligned[i]
                # Exit conditions: price < Donchian lower level OR stoploss hit
                if close_price < donchian_low_aligned[i] or close_price <= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                stop_loss = entry_price + 2.0 * atr_4h_aligned[i]
                # Exit conditions: price > Donchian upper level OR stoploss hit
                if close_price > donchian_high_aligned[i] or close_price >= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals
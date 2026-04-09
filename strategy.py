#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume confirmation and ADX regime filter
# - Uses 1d Donchian channels for breakout signals (long above 20-period high, short below 20-period low)
# - Confirms with 1w volume > 2.0x 20-period average (strong institutional participation)
# - Filters by 1d ADX > 25 (trending market) to avoid whipsaws in ranging conditions
# - Exits when price touches opposite Donchian level or ATR-based stoploss (2.0x ATR)
# - Position size: 0.25 (25% of capital) for balanced risk/return
# - Target: 15-30 trades/year on 1d timeframe (60-120 total over 4 years) to minimize fee drag
# - Works in bull markets (breakouts continue) and bear markets (breakdowns continue)
# - Donchian channels provide robust structure that adapts to volatility regimes
# - ADX filter ensures we only trade in trending conditions where breakouts are more reliable

name = "1d_1w_donchian_volume_adx_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute HTF indicators (1w)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # 1w Volume > 2.0x 20-period average (stricter for fewer trades)
    avg_volume_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_spike_1w = volume_1w > (2.0 * avg_volume_20_1w)
    
    # Align 1w volume spike to 1d
    volume_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_spike_1w.astype(float))
    
    # Pre-compute 1d indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 1d True Range for ATR and ADX
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # 1d ATR(14) for stoploss
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d ADX(14) for regime filter
    # +DM and -DM calculation
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed +DM, -DM, and TR
    tr_period = 14
    atr_for_adx = pd.Series(tr).rolling(window=tr_period, min_periods=tr_period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=tr_period, min_periods=tr_period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=tr_period, min_periods=tr_period).mean().values
    
    # +DI and -DI
    plus_di = 100 * (plus_dm_smooth / atr_for_adx)
    minus_di = 100 * (minus_dm_smooth / atr_for_adx)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=tr_period, min_periods=tr_period).mean().values
    
    # ADX > 25 indicates trending market
    adx_trending = adx > 25
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_spike_1w_aligned[i]) or np.isnan(adx_trending[i]) or
            np.isnan(atr_1d[i]) or atr_1d[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: opposite Donchian touch (low) or ATR stoploss
            if low[i] <= donchian_low[i]:  # Touch opposite band
                position = 0
                signals[i] = 0.0
            elif high[i] >= entry_price + (2.0 * atr_stop):  # ATR stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: opposite Donchian touch (high) or ATR stoploss
            if high[i] >= donchian_high[i]:  # Touch opposite band
                position = 0
                signals[i] = 0.0
            elif low[i] <= entry_price - (2.0 * atr_stop):  # ATR stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation and ADX trend filter
            if (high[i] >= donchian_high[i] and  # Break above upper band
                volume_spike_1w_aligned[i] and   # Volume confirmation
                adx_trending[i]):                # Trending market (ADX > 25)
                position = 1
                entry_price = high[i]
                atr_stop = atr_1d[i]
                signals[i] = 0.25
            elif (low[i] <= donchian_low[i] and    # Break below lower band
                  volume_spike_1w_aligned[i] and   # Volume confirmation
                  adx_trending[i]):                # Trending market (ADX > 25)
                position = -1
                entry_price = low[i]
                atr_stop = atr_1d[i]
                signals[i] = -0.25
    
    return signals
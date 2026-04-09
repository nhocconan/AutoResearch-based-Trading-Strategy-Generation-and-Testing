#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and ADX regime filter
# - Uses 4h Donchian channels for breakout signals (long above 20-period high, short below 20-period low)
# - Confirms with 12h volume > 2.0x 20-period average (strong institutional participation)
# - Filters by 12h ADX > 25 (trending market) to avoid choppy conditions
# - Exits when price touches opposite Donchian level or ATR-based stoploss (2.0x ATR)
# - Position size: 0.25 (25% of capital) for balanced risk/return
# - Target: 15-35 trades/year on 4h timeframe (60-140 total over 4 years) to minimize fee drag
# - Works in bull markets (breakouts continue) and bear markets (breakdowns continue)
# - Donchian channels provide robust structure that adapts to volatility regimes
# - ADX filter ensures we only trade in trending markets, reducing whipsaw

name = "4h_12h_donchian_volume_adx_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute HTF indicators
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12h True Range for ATR and ADX
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # 12h ATR(14) for stoploss
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 12h ADX(14) for regime filter
    plus_dm = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h), 
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    minus_dm = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)), 
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    tr_ma = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / np.where(tr_ma != 0, tr_ma, 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / np.where(tr_ma != 0, tr_ma, 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, (plus_di + minus_di), 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # 12h Donchian channels (20-period)
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # 12h Volume > 2.0x 20-period average (stricter for fewer trades)
    avg_volume_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_12h > (2.0 * avg_volume_20)
    
    # Align all 12h indicators to 4h
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike.astype(float))
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(atr_12h_aligned[i]) or atr_12h_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: opposite Donchian touch (low) or ATR stoploss
            if low[i] <= donchian_low_aligned[i]:  # Touch opposite band
                position = 0
                signals[i] = 0.0
            elif high[i] >= entry_price + (2.0 * atr_stop):  # ATR stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: opposite Donchian touch (high) or ATR stoploss
            if high[i] >= donchian_high_aligned[i]:  # Touch opposite band
                position = 0
                signals[i] = 0.0
            elif low[i] <= entry_price - (2.0 * atr_stop):  # ATR stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation and ADX > 25 (trending market)
            if (high[i] >= donchian_high_aligned[i] and  # Break above upper band
                volume_spike_aligned[i] and         # Volume confirmation
                adx_aligned[i] > 25):               # Trending market filter
                position = 1
                entry_price = high[i]
                atr_stop = atr_12h_aligned[i]
                signals[i] = 0.25
            elif (low[i] <= donchian_low_aligned[i] and   # Break below lower band
                  volume_spike_aligned[i] and         # Volume confirmation
                  adx_aligned[i] > 25):               # Trending market filter
                position = -1
                entry_price = low[i]
                atr_stop = atr_12h_aligned[i]
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ADX trend filter and volume confirmation
# Donchian breakout captures strong momentum moves in direction of 1d trend (ADX > 25)
# Volume confirmation (1.5x 20-period average) ensures institutional participation
# Discrete sizing 0.25 targets 50-150 trades over 4 years (12-37/year)
# Works in bull/bear by only taking breakouts in direction of 1d trend

name = "12h_Donchian20_1dADX25_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on 1d
    plus_dm = pd.Series(df_1d['high']).diff()
    minus_dm = pd.Series(df_1d['low']).diff().copy()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = abs(minus_dm)
    
    tr1 = pd.Series(df_1d['high']) - pd.Series(df_1d['low'])
    tr2 = abs(pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift(1))
    tr3 = abs(pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=14, min_periods=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (abs(plus_di + minus_di))) * 100
    adx = dx.rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 12h timeframe (completed 1d bar only)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Donchian channels (20-period) on 12h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for calculations)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(adx_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: price > 20-period high with 1d uptrend (ADX > 25 and +DI > -DI)
            long_breakout = close[i] > highest_high[i]
            # Short breakdown: price < 20-period low with 1d downtrend (ADX > 25 and -DI > +DI)
            short_breakout = close[i] < lowest_low[i]
            
            # 1d ADX trend filter: ADX > 25 indicates strong trend
            adx_strong = adx_aligned[i] > 25
            # Get 1d DI values for trend direction
            plus_dm_1d = pd.Series(df_1d['high']).diff()
            minus_dm_1d = pd.Series(df_1d['low']).diff().copy()
            plus_dm_1d[plus_dm_1d < 0] = 0
            minus_dm_1d[minus_dm_1d > 0] = 0
            minus_dm_1d = abs(minus_dm_1d)
            tr1_1d = pd.Series(df_1d['high']) - pd.Series(df_1d['low'])
            tr2_1d = abs(pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift(1))
            tr3_1d = abs(pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift(1))
            tr_1d = pd.concat([tr1_1d, tr2_1d, tr3_1d], axis=1).max(axis=1)
            atr_1d = tr_1d.rolling(window=14, min_periods=14).mean()
            plus_di_1d = 100 * (plus_dm_1d.rolling(window=14, min_periods=14).mean() / atr_1d)
            minus_di_1d = 100 * (minus_dm_1d.rolling(window=14, min_periods=14).mean() / atr_1d)
            plus_di_1d_aligned = align_htf_to_ltf(prices, df_1d, plus_di_1d.values)
            minus_di_1d_aligned = align_htf_to_ltf(prices, df_1d, minus_di_1d.values)
            
            adx_long = adx_strong and (plus_di_1d_aligned[i] > minus_di_1d_aligned[i])
            adx_short = adx_strong and (minus_di_1d_aligned[i] > plus_di_1d_aligned[i])
            
            if long_breakout and adx_long and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            elif short_breakout and adx_short and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < 20-period low or ADX weakens (< 20)
            if close[i] < lowest_low[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price > 20-period high or ADX weakens (< 20)
            if close[i] > highest_high[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
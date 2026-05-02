#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d ADX trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions for mean reversion entries
# 1d ADX > 25 ensures trades align with strong daily trend to avoid choppy whipsaws
# Volume spike (1.5x 20-period average) confirms participation
# Discrete sizing 0.25 targets 50-150 trades over 4 years (12-37/year)
# Works in bull/bear by only taking reversals in direction of 1d trend

name = "6h_WilliamsR_1dADX25_Trend_Volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Williams %R(14) on 1d: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - df_1d['close'].values) / (highest_high - lowest_low)) * -100
    
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
    
    # Align Williams %R and ADX to 6h timeframe (completed 1d bar only)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for calculations)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Williams %R oversold long: < -80 with 1d uptrend (ADX > 25 and +DI > -DI)
            williams_long = williams_r_aligned[i] < -80
            # Williams %R overbought short: > -20 with 1d downtrend (ADX > 25 and -DI > +DI)
            williams_short = williams_r_aligned[i] > -20
            
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
            
            if williams_long and adx_long and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            elif williams_short and adx_short and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R overbought (> -20) or ADX weakens (< 20)
            if williams_r_aligned[i] > -20 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R oversold (< -80) or ADX weakens (< 20)
            if williams_r_aligned[i] < -80 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
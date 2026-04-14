#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R for overbought/oversold signals and 1w ADX for trend strength.
# Williams %R identifies reversal points in ranging markets (Williams %R > -20 = overbought, < -80 = oversold).
# ADX > 25 indicates strong trend where we follow momentum; ADX < 25 indicates ranging where we fade extremes.
# In ranging markets (ADX < 25): short at Williams %R > -20, long at Williams %R < -80.
# In trending markets (ADX >= 25): long when Williams %R crosses above -50, short when crosses below -50.
# Volume confirmation required for all entries to avoid false signals.
# Designed to work in both bull (trend following) and bear (mean reversion in ranges) markets.
# Target: 25-35 trades/year per symbol (100-140 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R(14) on 1d
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Load 1w data ONCE for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX(14) on 1w
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = np.concatenate([[np.nan], high_1w[1:] - high_1w[:-1]])
    down_move = np.concatenate([[np.nan], low_1w[:-1] - low_1w[1:]])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr_1w = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di_1w = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1w
    minus_di_1w = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1w
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w)
    dx = np.where((plus_di_1w + minus_di_1w) == 0, 0, dx)
    adx_1w = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align indicators to lower timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Volume confirmation: 1.3x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 14, 14)  # Need Williams %R, ADX, and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(adx_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # Market regime: ADX >= 25 = trending, ADX < 25 = ranging
        trending = adx_1w_aligned[i] >= 25
        ranging = adx_1w_aligned[i] < 25
        
        if position == 0:
            if ranging:
                # In ranging markets: fade extremes
                # Long when oversold (Williams %R < -80)
                if (williams_r_aligned[i] < -80 and volume_confirmed):
                    position = 1
                    signals[i] = position_size
                # Short when overbought (Williams %R > -20)
                elif (williams_r_aligned[i] > -20 and volume_confirmed):
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            else:  # trending
                # In trending markets: follow momentum
                # Long when Williams %R crosses above -50 (momentum building)
                if (williams_r_aligned[i] > -50 and 
                    williams_r_aligned[i-1] <= -50 and 
                    volume_confirmed):
                    position = 1
                    signals[i] = position_size
                # Short when Williams %R crosses below -50 (momentum weakening)
                elif (williams_r_aligned[i] < -50 and 
                      williams_r_aligned[i-1] >= -50 and 
                      volume_confirmed):
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to overbought territory or ADX drops indicating trend weakening
            if (williams_r_aligned[i] > -20 or 
                adx_1w_aligned[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns to oversold territory or ADX drops indicating trend weakening
            if (williams_r_aligned[i] < -80 or 
                adx_1w_aligned[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_WilliamsR_ADX_Regime_v1"
timeframe = "6h"
leverage = 1.0
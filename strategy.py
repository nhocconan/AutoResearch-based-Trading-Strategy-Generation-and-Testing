#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout + 1w volume spike + ADX regime filter
# - 12h Camarilla levels (H3/L3) from 1d provide institutional support/resistance
# - 1w volume > 1.5x 20-week average confirms strong participation
# - ADX(14) > 20 on 1w filters for trending regimes (avoid false breakouts in chop)
# - Discrete position sizing (0.25) minimizes fee churn
# - Target: 12-30 trades/year (50-120 total over 4 years) to avoid fee drag
# - Works in bull/bear: breakouts capture trends, volume confirms legitimacy, ADX avoids chop

name = "12h_1w_camarilla_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d True Range for pivot
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_1d[0] = np.nan
    tr2_1d[0] = np.nan
    tr3_1d[0] = np.nan
    tr_1d = np.maximum.reduce([tr1_1d, tr2_1d, tr3_1d])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels (based on previous day)
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 6
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 6
    
    # Align Camarilla levels to 12h
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Pre-compute 1w volume and ADX
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # 1w ATR for ADX calculation
    tr1_1w = high_1w - low_1w
    tr2_1w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3_1w = np.abs(low_1w - np.roll(close_1w, 1))
    tr1_1w[0] = np.nan
    tr2_1w[0] = np.nan
    tr3_1w[0] = np.nan
    tr_1w = np.maximum.reduce([tr1_1w, tr2_1w, tr3_1w])
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    # +DM and -DM for ADX
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    up_move[0] = np.nan
    down_move[0] = np.nan
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and TR for ADX
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    tr_smooth = pd.Series(tr_1w).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI and ADX
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align HTF indicators to 12h
    volume_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # 1w volume MA (20-period)
    volume_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_20_1w)
    
    # Pre-compute 12h close
    close_12h = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(volume_1w_aligned[i]) or np.isnan(volume_ma_20_aligned[i]) or 
            np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1w volume > 1.5x 20-week average
        volume_confirm = volume_1w_aligned[i] > 1.5 * volume_ma_20_aligned[i]
        
        # Regime filter: ADX > 20 (trending market)
        regime_filter = adx_1w_aligned[i] > 20
        
        close_price = close_12h[i]
        
        # Camarilla breakout conditions
        long_breakout = close_price > camarilla_h3_aligned[i]
        short_breakout = close_price < camarilla_l3_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require both volume confirmation and regime filter
            if long_breakout and volume_confirm and regime_filter:
                position = 1
                signals[i] = 0.25
            elif short_breakout and volume_confirm and regime_filter:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit on opposite Camarilla level touch or ADX weakening
            if position == 1:
                exit_condition = (close_price < camarilla_l3_aligned[i]) or (adx_1w_aligned[i] < 15)
            else:  # position == -1
                exit_condition = (close_price > camarilla_h3_aligned[i]) or (adx_1w_aligned[i] < 15)
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d volume spike and ADX regime filter
# - Uses Williams Alligator (3 SMAs: Jaw=13, Teeth=8, Lips=5) on 12h timeframe
# - Long when Lips > Teeth > Jaw (bullish alignment) with volume confirmation
# - Short when Lips < Teeth < Jaw (bearish alignment) with volume confirmation
# - ADX(14) on 1d timeframe filters for trending markets (ADX > 20) to avoid false signals in ranging conditions
# - Volume confirmation: current 12h volume > 1.5x 20-period average
# - Discrete position sizing (0.25) minimizes fee churn
# - Target: 15-40 trades/year (60-160 total over 4 years) to stay within HARD MAX: 400 total
# - Williams Alligator excels in trending markets and avoids whipsaws in ranging conditions
# - Works in both bull and bear markets by capturing strong trends while filtering noise

name = "12h_1d_williams_alligator_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d OHLC for ADX calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pre-compute 1d ADX(14) for regime filter
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum.reduce([tr1, tr2, tr3])
    
    # +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = np.nan
    down_move[0] = np.nan
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    tr_period = 14
    atr_1d = pd.Series(tr).rolling(window=tr_period, min_periods=tr_period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=tr_period, min_periods=tr_period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=tr_period, min_periods=tr_period).mean().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / atr_1d
    minus_di = 100 * minus_dm_smooth / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).rolling(window=tr_period, min_periods=tr_period).mean().values
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Pre-compute 12h Williams Alligator
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Median price = (High + Low) / 2
    median_price = (high_12h + low_12h) / 2.0
    
    # Williams Alligator lines (all SMAs)
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    jaw = pd.Series(median_price).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    teeth = pd.Series(median_price).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    lips = pd.Series(median_price).rolling(window=lips_period, min_periods=lips_period).mean().values
    
    # Pre-compute 12h volume and its 20-period moving average
    volume_12h = prices['volume'].values
    volume_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(volume_ma_20_12h[i]) or np.isnan(adx_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Williams Alligator conditions
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        volume_confirm = volume_12h[i] > 1.5 * volume_ma_20_12h[i]
        
        # Regime filter: ADX > 20 indicates trending market
        trending_market = adx_aligned[i] > 20.0
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: bullish Alligator alignment AND volume confirmation AND trending market
            if bullish_alignment and volume_confirm and trending_market:
                position = 1
                signals[i] = 0.25
            # Short conditions: bearish Alligator alignment AND volume confirmation AND trending market
            elif bearish_alignment and volume_confirm and trending_market:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when Alligator alignment breaks (Lips crosses Teeth)
            if position == 1 and lips[i] <= teeth[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and lips[i] >= teeth[i]:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals
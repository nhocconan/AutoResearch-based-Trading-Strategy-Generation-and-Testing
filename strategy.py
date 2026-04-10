#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + Williams Alligator Combination
# - Primary: 6h timeframe for balanced trade frequency and reduced fee drag
# - HTF: 12h for trend direction (ADX > 25) and 1d for Alligator alignment
# - Long: 12h ADX > 25 + price > 6h Alligator (teeth) + 6h close > 1d Alligator jaws
# - Short: 12h ADX > 25 + price < 6h Alligator (teeth) + 6h close < 1d Alligator jaws
# - Exit: ADX < 20 (trend weakening) or price crosses 8-period EMA on 6h
# - Position sizing: 0.25 (discrete level)
# - Target: 50-150 total trades over 4 years (12-37/year) - within 6h sweet spot
# - Works in bull/bear: ADX filters ranging markets, Alligator catches trends in both directions

name = "6h_12h_1d_adx_alligator_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 12h data for ADX
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Pre-compute 1d data for Alligator
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12h ADX(14)
    # True Range
    tr1 = pd.Series(high_12h).shift(1) - pd.Series(low_12h).shift(1)
    tr2 = abs(pd.Series(high_12h) - pd.Series(close_12h).shift(1))
    tr3 = abs(pd.Series(low_12h) - pd.Series(close_12h).shift(1))
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = pd.Series(high_12h) - pd.Series(high_12h).shift(1)
    down_move = pd.Series(low_12h).shift(1) - pd.Series(low_12h)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Calculate 6h Alligator (Jaw=13, Teeth=8, Lips=5 SMAs of median price)
    median_6h = (high_6h + low_6h) / 2.0
    jaw_6h = pd.Series(median_6h).rolling(window=13, min_periods=13).mean().values
    teeth_6h = pd.Series(median_6h).rolling(window=8, min_periods=8).mean().values
    lips_6h = pd.Series(median_6h).rolling(window=5, min_periods=5).mean().values
    
    # Calculate 1d Alligator
    median_1d = (high_1d + low_1d) / 2.0
    jaw_1d = pd.Series(median_1d).rolling(window=13, min_periods=13).mean().values
    teeth_1d = pd.Series(median_1d).rolling(window=8, min_periods=8).mean().values
    lips_1d = pd.Series(median_1d).rolling(window=5, min_periods=5).mean().values
    
    # Align HTF indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Calculate 6h 8-period EMA for exit signal
    ema_8_6h = pd.Series(close_6h).ewm(span=8, adjust=False, min_periods=8).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(jaw_6h[i]) or np.isnan(teeth_6h[i]) or
            np.isnan(jaw_1d_aligned[i]) or np.isnan(ema_8_6h[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions (require strong trend)
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Strong trend + price above 6h teeth + close above 1d jaw
            if (strong_trend and 
                close_6h[i] > teeth_6h[i] and 
                close_6h[i] > jaw_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Strong trend + price below 6h teeth + close below 1d jaw
            elif (strong_trend and 
                  close_6h[i] < teeth_6h[i] and 
                  close_6h[i] < jaw_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Trend weakening (ADX < 20)
            # 2. Price crosses 8-period EMA (mean reversion signal)
            
            if position == 1:  # Long position
                exit_condition = (
                    adx_aligned[i] < 20 or  # Trend weakening
                    close_6h[i] < ema_8_6h[i]  # Price below 8 EMA
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (
                    adx_aligned[i] < 20 or  # Trend weakening
                    close_6h[i] > ema_8_6h[i]  # Price above 8 EMA
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals
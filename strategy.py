#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d ADX trend filter and volume spike confirmation.
# Long when Williams %R crosses above -80 (oversold bounce) AND 1d ADX > 25 (trending market) AND volume > 1.5x 20-bar average.
# Short when Williams %R crosses below -20 (overbought rejection) AND 1d ADX > 25 AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to balance profit potential and drawdown control.
# Williams %R identifies exhaustion points in trending markets, ADX filters for sufficient trend strength,
# volume spike confirms conviction. Designed for 60-120 total trades over 4 years (15-30/year).

name = "6h_WilliamsR_ADX_VolumeSpike_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(abs(high_1d - pd.Series(close_1d).shift(1)))
    tr3 = pd.Series(abs(low_1d - pd.Series(close_1d).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1d - pd.Series(high_1d).shift(1))
    down_move = pd.Series(pd.Series(low_1d).shift(1) - low_1d)
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)).values
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)).values
    
    # Directional Indicators
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    dx = 100 * abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Williams %R (14-period) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume confirmation: current 6h volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for Williams %R and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(williams_r[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if vol_ma[i] <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = volume[i] > (vol_ma[i] * 1.5)
        strong_trend = adx_1d_aligned[i] > 25
        
        # Williams %R signals: crossover above -80 (long) or below -20 (short)
        williams_long_signal = williams_r[i] > -80 and williams_r[i-1] <= -80
        williams_short_signal = williams_r[i] < -20 and williams_r[i-1] >= -20
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 AND strong trend AND volume confirmation
            if williams_long_signal and strong_trend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 AND strong trend AND volume confirmation
            elif williams_short_signal and strong_trend and volume_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses below -50 (momentum loss) OR trend weakens
            if williams_r[i] < -50 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses above -50 (momentum loss) OR trend weakens
            if williams_r[i] > -50 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
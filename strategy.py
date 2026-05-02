#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Reversal with 1d ADX trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions (%R < -80 oversold, %R > -20 overbought)
# 1d ADX > 25 confirms strong trend to avoid choppy markets and false reversals
# Volume spike (>1.5 x 20-period EMA) validates reversal strength
# Works in bull markets (oversold bounce in uptrend) and bear markets (overbought rejection in downtrend)
# Uses discrete position sizing (0.25) to minimize fee churn and control drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag

name = "6h_WilliamsR_Reversal_1dADX_Trend_VolumeConfirmation"
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
    
    # 6h Williams %R (14-period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d ADX calculation (14-period)
    plus_dm = pd.Series(df_1d['high']).diff()
    minus_dm = pd.Series(df_1d['low']).diff().mul(-1)
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr1 = pd.Series(df_1d['high']).subtract(df_1d['low'])
    tr2 = pd.Series(df_1d['high']).subtract(df_1d['close'].shift(1)).abs()
    tr3 = pd.Series(df_1d['low']).subtract(df_1d['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (abs(plus_di + minus_di))) * 100
    adx = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation (volume spike > 1.5 x 20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Williams %R and ADX calculation)
    start_idx = 30
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d ADX (strong trend when ADX > 25)
        strong_uptrend = adx_aligned[i] > 25 and close[i] > close[i-1]  # ADX > 25 + price rising
        strong_downtrend = adx_aligned[i] > 25 and close[i] < close[i-1]  # ADX > 25 + price falling
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R oversold (< -80) with volume confirmation and strong uptrend
            if williams_r[i] < -80 and volume_confirmation[i] and strong_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) with volume confirmation and strong downtrend
            elif williams_r[i] > -20 and volume_confirmation[i] and strong_downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R rises above -50 (momentum weakening) OR ADX falls below 20 (trend weakening)
            if williams_r[i] > -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R falls below -50 (momentum weakening) OR ADX falls below 20 (trend weakening)
            if williams_r[i] < -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
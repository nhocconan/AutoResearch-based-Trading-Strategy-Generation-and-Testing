#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Reversal with 1d ADX trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions: Long when %R crosses above -80 from below,
# Short when %R crosses below -20 from above. 1d ADX > 25 ensures we only trade in strong trends
# to avoid false reversals in ranging markets. Volume spike (>1.5 x 20-period EMA) confirms
# reversal validity. Works in bull markets (buy oversold in uptrend) and bear markets
# (sell overbought in downtrend). Uses discrete position sizing (0.25) to minimize fee churn
# and control drawdown. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

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
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # 1d data for trend filter (ADX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 1d ADX calculation (14-period)
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
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation (volume spike > 1.5 x 20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for indicator calculation)
    start_idx = 34
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d ADX
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 from below (oversold bounce) in strong trend
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and strong_trend and volume_confirmation[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above (overbought rejection) in strong trend
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and strong_trend and volume_confirmation[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses above -20 (overbought) OR trend weakens
            if williams_r[i] >= -20 or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -80 (oversold) OR trend weakens
            if williams_r[i] <= -80 or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
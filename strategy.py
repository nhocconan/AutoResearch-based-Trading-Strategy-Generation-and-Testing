#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R Reversal with 1w EMA34 trend filter and volume spike confirmation
# Williams %R identifies overbought/oversold conditions; reversals from extreme levels
# capture mean reversion in ranging markets while avoiding strong trends via 1w EMA filter
# Volume spike (>2.0 x 30-period EMA) confirms reversal validity and reduces false signals
# Discrete position sizing (0.25) balances opportunity with fee drag control
# Target: 30-100 total trades over 4 years (7-25/year) for optimal risk-adjusted returns

name = "1d_WilliamsR_Reversal_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation (volume spike > 2.0 x 30-period EMA)
    vol_ema_30 = pd.Series(volume).ewm(span=30, adjust=False, min_periods=30).mean().values
    volume_confirmation = volume > (2.0 * vol_ema_30)
    
    # Williams %R (14-period): (Highest High - Close) / (Highest High - Lowest Low) * -100
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid div by zero
    
    # 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for calculations)
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1w EMA34
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 (oversold reversal) with volume confirmation and not in strong uptrend
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and volume_confirmation[i] and not uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought reversal) with volume confirmation and not in strong downtrend
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and volume_confirmation[i] and not downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses below -50 (momentum loss) OR trend changes to uptrend (avoid counter-trend)
            if williams_r[i] < -50 or uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses above -50 (momentum loss) OR trend changes to downtrend (avoid counter-trend)
            if williams_r[i] > -50 or downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
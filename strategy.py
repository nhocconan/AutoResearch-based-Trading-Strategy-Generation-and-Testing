#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R with 1w EMA50 trend filter and volume confirmation
# Williams %R measures overbought/oversold levels: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
# Long when %R < -80 (oversold) with 1w EMA50 uptrend and volume spike
# Short when %R > -20 (overbought) with 1w EMA50 downtrend and volume spike
# 1w EMA50 provides higher-timeframe trend alignment to avoid counter-trend trades
# Volume confirmation (>2.0 x 20-period EMA) filters weak breakouts
# Works in bull markets (oversold bounces in uptrend) and bear markets (overbought rejections in downtrend)
# Uses discrete position sizing (0.25) to minimize fee churn and control drawdown
# Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag

name = "1d_WilliamsR_1wEMA50_Trend_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 calculation
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation (volume spike > 2.0 x 20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Williams %R and EMA calculation)
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R < -80 (oversold) with volume confirmation and uptrend
            if williams_r[i] < -80 and volume_confirmation[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) with volume confirmation and downtrend
            elif williams_r[i] > -20 and volume_confirmation[i] and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R > -50 (exiting oversold) OR trend changes to downtrend
            if williams_r[i] > -50 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R < -50 (exiting overbought) OR trend changes to uptrend
            if williams_r[i] < -50 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Williams %R reversal with 1d EMA50 trend filter and volume confirmation
    # Williams %R identifies overbought/oversold conditions; reversals from extremes capture mean reversion
    # Works in both bull and bear markets: buys oversold dips in uptrends, sells overbought rallies in downtrends
    # Volume surge confirms reversal strength, EMA50 filters trend direction to avoid counter-trend trades
    
    # Load daily data once
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily EMA50 trend filter
    ema_1d_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # 4h Williams %R (14-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R: -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume filter (20-period MA surge)
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_50_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold reversal) with volume surge AND daily EMA50 uptrend
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and vol_surge[i] and close[i] > ema_1d_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought reversal) with volume surge AND daily EMA50 downtrend
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and vol_surge[i] and close[i] < ema_1d_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to neutral zone (-50) or opposite extreme
            if position == 1:
                if williams_r[i] < -50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if williams_r[i] > -50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_Reversal_1dEMA50_Trend_VolumeSurge_v1"
timeframe = "4h"
leverage = 1.0
#!/usr/bin/env python3
# Hypothesis: 6h Williams %R with 12h EMA50 trend filter and volume spike confirmation.
# Williams %R(14) identifies overbought/oversold conditions. 
# Long when Williams %R crosses above -80 (exiting oversold) with volume > 1.5x average AND price > 12h EMA50.
# Short when Williams %R crosses below -20 (exiting overbought) with volume > 1.5x average AND price < 12h EMA50.
# Exit on opposite Williams %R level (-20 for long, -80 for short) or trend reversal.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year.
# Works in bull markets via buying oversold dips in uptrend and in bear markets via selling overbought rallies in downtrend.
# Williams %R is mean-reverting but with trend filter avoids counter-trend whipsaws.
# 6h timeframe reduces trade frequency vs lower TFs, improving fee drag profile.

name = "6h_WilliamsR_12hTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R(14) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    highest_high = high_series.rolling(window=14, min_periods=14).max()
    lowest_low = low_series.rolling(window=14, min_periods=14).min()
    williams_r = (highest_high - close_series) / (highest_high - lowest_low) * -100
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA(50) on 12h close for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R crosses above -80 (exiting oversold) with volume confirmation AND price > 12h EMA50
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and volume_filter[i] and close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R crosses below -20 (exiting overbought) with volume confirmation AND price < 12h EMA50
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and volume_filter[i] and close[i] < ema50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses below -20 (overbought) OR trend reversal (price < 12h EMA50)
            if williams_r[i] < -20 and williams_r[i-1] >= -20 or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses above -80 (oversold) OR trend reversal (price > 12h EMA50)
            if williams_r[i] > -80 and williams_r[i-1] <= -80 or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
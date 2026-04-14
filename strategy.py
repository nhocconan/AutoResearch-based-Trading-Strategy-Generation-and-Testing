#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d trend filter and volume confirmation
# Targets: 30-60 trades/year by requiring oversold/overbought conditions during pullbacks in trend
# Logic: Long when Williams %R(14) < -80 (oversold) and price > 1d EMA50 (uptrend) with volume > 1.5x average
#        Short when Williams %R(14) > -20 (overbought) and price < 1d EMA50 (downtrend) with volume > 1.5x average
#        Uses Williams %R for mean reversion within trend, EMA50 for trend direction, volume for confirmation
# Position size: 0.25 to manage drawdown

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Williams %R (14) calculation
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(50, n):
        # Get aligned daily EMA50
        ema_50_i = align_htf_to_ltf(prices, df_1d, ema_50_1d)[i]
        
        if np.isnan(williams_r[i]) or np.isnan(ema_50_i) or np.isnan(vol_ma[i]):
            continue
        
        # Volume confirmation (1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Long: Williams %R oversold + uptrend + volume confirmation
        if position == 0 and williams_r[i] < -80 and close[i] > ema_50_i and volume_confirm:
            position = 1
            signals[i] = position_size
        # Short: Williams %R overbought + downtrend + volume confirmation
        elif position == 0 and williams_r[i] > -20 and close[i] < ema_50_i and volume_confirm:
            position = -1
            signals[i] = -position_size
        # Exit: Williams %R returns to neutral zone (-50) or opposite signal
        elif position != 0:
            if position == 1 and (williams_r[i] > -50 or williams_r[i] > -20):
                position = 0
                signals[i] = 0.0
            elif position == -1 and (williams_r[i] < -50 or williams_r[i] < -80):
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_WilliamsR_MeanReversion_TrendFilter_Volume"
timeframe = "4h"
leverage = 1.0
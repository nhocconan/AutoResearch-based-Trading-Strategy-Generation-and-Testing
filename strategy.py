#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d RSI trend filter and volume confirmation.
# Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
# Long when Williams %R crosses above -80 (oversold) AND 1d RSI > 50 (uptrend) AND volume > 1.2x average
# Short when Williams %R crosses below -20 (overbought) AND 1d RSI < 50 (downtrend) AND volume > 1.2x average
# Exit when Williams %R crosses opposite threshold or RSI flips direction.
# Designed for mean reversion in ranging markets with trend filter to avoid counter-trend trades.
# Volume filter ensures participation. Target: 20-40 trades/year to avoid fee drag.
name = "6h_WilliamsR_1dRSI_VolumeFilter"
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
    
    # Williams %R (14-period)
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    
    # Williams %R cross signals
    williams_r_above_80 = np.zeros(n, dtype=bool)  # Williams %R > -80 (oversold exit)
    williams_r_below_20 = np.zeros(n, dtype=bool)  # Williams %R < -20 (overbought exit)
    williams_r_cross_up_80 = np.zeros(n, dtype=bool)   # Cross above -80
    williams_r_cross_down_20 = np.zeros(n, dtype=bool) # Cross below -20
    
    williams_r_above_80[1:] = (williams_r[1:] > -80) & (williams_r[:-1] <= -80)
    williams_r_below_20[1:] = (williams_r[1:] < -20) & (williams_r[:-1] >= -20)
    williams_r_cross_up_80[1:] = (williams_r[1:] > -80) & (williams_r[:-1] <= -80)
    williams_r_cross_down_20[1:] = (williams_r[1:] < -20) & (williams_r[:-1] >= -20)
    
    # 1d RSI for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # RSI calculation using Wilder's smoothing
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = np.where(avg_loss == 0, 100, rsi_1d)  # Handle division by zero
    
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # RSI trend filters
    rsi_above_50 = rsi_1d_aligned > 50
    rsi_below_50 = rsi_1d_aligned < 50
    
    # Volume filter: current volume > 1.2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.2 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma20[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 (from oversold) AND RSI > 50 (uptrend) AND volume filter
            long_cond = williams_r_cross_up_80[i] and rsi_above_50[i] and volume_filter[i]
            # Short: Williams %R crosses below -20 (from overbought) AND RSI < 50 (downtrend) AND volume filter
            short_cond = williams_r_cross_down_20[i] and rsi_below_50[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -20 (overbought) OR RSI < 50 (trend change) OR volume filter fails
            if williams_r_cross_down_20[i] or (not rsi_above_50[i]) or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -80 (oversold) OR RSI > 50 (trend change) OR volume filter fails
            if williams_r_cross_up_80[i] or (not rsi_below_50[i]) or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
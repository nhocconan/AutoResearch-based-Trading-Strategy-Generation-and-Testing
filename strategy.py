#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 12h EMA34 trend filter and 1d volume confirmation.
# Long when Williams %R(14) < -80 (oversold) AND 12h close > EMA34 (uptrend) AND 1d volume > 1.5x 20-period volume MA.
# Short when Williams %R(14) > -20 (overbought) AND 12h close < EMA34 (downtrend) AND 1d volume > 1.5x 20-period volume MA.
# Williams %R identifies exhaustion points in ranging markets, 12h EMA34 filters for higher-timeframe trend alignment,
# and 1d volume confirms institutional participation at reversal points. Designed for 6h timeframe to achieve
# 50-150 total trades over 4 years (12-37/year) with strict entry conditions. Works in both bull and bear markets
# by trading mean reversions in the direction of the 12h trend when volume confirms, avoiding counter-trend whipsaws.

name = "6h_WilliamsR_Extreme_12hEMA34_1dVolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC) to avoid datetime64 issues
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend direction
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume 20-period MA for spike detection
    volume_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Calculate 6h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(volume_ma_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        wr_val = williams_r[i]
        
        # Volume spike condition: current 6h volume > 2.0x 20-period 6h volume MA (proxy for 1d volume spike)
        volume_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume[i] > (volume_ma_6h[i] * 2.0)
        
        # Williams %R extreme conditions
        oversold = wr_val < -80.0
        overbought = wr_val > -20.0
        
        # 12h trend conditions
        trend_up = close_val > ema_34_12h_aligned[i]
        trend_down = close_val < ema_34_12h_aligned[i]
        
        if position == 0:
            # Long: Williams %R oversold AND 12h uptrend AND volume spike AND session
            if oversold and trend_up and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought AND 12h downtrend AND volume spike AND session
            elif overbought and trend_down and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -50 (mean reversion) OR trend changes
            if wr_val > -50.0 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -50 (mean reversion) OR trend changes
            if wr_val < -50.0 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
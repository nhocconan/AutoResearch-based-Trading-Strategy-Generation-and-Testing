#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R extreme with 1d EMA34 trend filter and volume spike.
# Long when Williams %R(14) < -80 (oversold) AND 1d close > 1d EMA34 (uptrend) AND 4h volume > 1.5x 20-period volume MA.
# Short when Williams %R(14) > -20 (overbought) AND 1d close < 1d EMA34 (downtrend) AND 4h volume > 1.5x 20-period volume MA.
# Exit when Williams %R crosses above -50 (for long) or below -50 (for short) OR trend reverses.
# Uses session filter (08-20 UTC) to avoid low-liquidity periods. Position size 0.25.
# Williams %R identifies exhaustion points; trading extreme readings in trend direction with volume confirmation
# provides mean-reversion entries within the trend, reducing whipsaw vs pure breakout strategies.
# Effective in both bull and bear markets by only taking counter-trend exhaustion trades aligned with higher-timeframe trend.

name = "4h_WilliamsR_Extreme_1dEMA34_VolumeSpike_Session"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend direction
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R(14) on 4h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 4h volume 20-period MA for spike detection
    volume_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_ma_4h[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        wr_val = williams_r[i]
        
        # Volume spike condition: current 4h volume > 1.5x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_4h[i] * 1.5)
        
        # Williams %R conditions
        oversold = wr_val < -80   # Oversold condition
        overbought = wr_val > -20  # Overbought condition
        wr_cross_up = wr_val > -50  # Williams %R crosses above -50 (exit long)
        wr_cross_down = wr_val < -50  # Williams %R crosses below -50 (exit short)
        
        # 1d trend conditions
        trend_up = close_val > ema_34_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_34_1d_aligned[i]  # 1d downtrend
        
        if position == 0:
            # Long: Williams %R oversold AND 1d uptrend AND volume spike AND session
            if oversold and trend_up and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought AND 1d downtrend AND volume spike AND session
            elif overbought and trend_down and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -50 OR trend changes
            if wr_cross_up or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -50 OR trend changes
            if wr_cross_down or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
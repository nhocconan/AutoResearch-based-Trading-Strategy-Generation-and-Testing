#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1w EMA50 trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold) in bull trend (close > 1w EMA50) with volume > 1.8x 24-period MA.
# Short when Williams %R > -20 (overbought) in bear trend (close < 1w EMA50) with volume spike.
# Uses discrete position sizing (0.25) to minimize fee churn. 1w EMA50 provides strong trend filter.
# Volume confirmation ensures institutional participation. Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_WilliamsR_1wEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Williams %R calculation (14-period)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R for each 1d bar: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume regime: current 12h volume > 1.8x 24-period MA
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.8 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_1w_aligned[i]
        wr_value = williams_r_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Williams %R conditions
        is_oversold = wr_value < -80
        is_overbought = wr_value > -20
        
        # Entry logic
        if position == 0:
            if is_bull_trend and is_oversold and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_bear_trend and is_overbought and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below EMA50 OR Williams %R becomes overbought
            if close_val < ema_trend or wr_value > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above EMA50 OR Williams %R becomes oversold
            if close_val > ema_trend or wr_value < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
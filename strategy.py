#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 12h EMA34 trend filter and volume confirmation
# Long when Williams %R < -80 (oversold), price > 12h EMA34 (uptrend), volume > 1.5x average
# Short when Williams %R > -20 (overbought), price < 12h EMA34 (downtrend), volume > 1.5x average
# Exit when Williams %R crosses -50 (mean reversion midpoint)
# Uses discrete position sizing (0.25) to target 12-30 trades/year on 6h timeframe.
# Williams %R is effective in ranging markets and the 12h EMA filter ensures we only trade with the higher timeframe trend,
# making it suitable for both bull and bear regimes by avoiding counter-trend trades.

name = "6h_WilliamsR_12hEMA34_VolumeMeanReversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 6h data for Williams %R (14-period)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 14:
        return np.zeros(n)
    
    # Calculate 6h Williams %R: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_6h['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_6h['low'].values).rolling(window=14, min_periods=14).min().values
    close_6h = df_6h['close'].values
    williams_r = (highest_high - close_6h) / (highest_high - lowest_low) * -100
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align 6h Williams %R to 6h timeframe (no additional delay needed for Williams %R)
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 34, 20)  # Warmup for Williams %R, 12h EMA34, and volume MA
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_williams_r = williams_r_aligned[i]
        curr_ema34_12h = ema_34_12h_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Williams %R crosses above -50 (mean reversion to midpoint)
            if curr_williams_r > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -50 (mean reversion to midpoint)
            if curr_williams_r < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirmed = curr_volume > 1.5 * curr_vol_ma
            
            # Long when Williams %R < -80 (oversold), price > 12h EMA34 (uptrend), volume confirmed
            if curr_williams_r < -80 and curr_close > curr_ema34_12h and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when Williams %R > -20 (overbought), price < 12h EMA34 (downtrend), volume confirmed
            elif curr_williams_r > -20 and curr_close < curr_ema34_12h and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals
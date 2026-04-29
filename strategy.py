#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d EMA34 trend filter and volume confirmation
# Long when Williams %R crosses above -80 (oversold), price > 1d EMA34, volume > 1.3x average
# Short when Williams %R crosses below -20 (overbought), price < 1d EMA34, volume > 1.3x average
# Exit when Williams %R crosses -50 (mean reversion)
# Uses discrete position sizing (0.25) and moderate volume filter to target ~50-100 trades/year.
# Williams %R captures reversals in ranging markets, EMA34 filter avoids counter-trend trades.
# Designed to work in both bull and bear markets by following the 1d trend.

name = "6h_WilliamsR_1dEMA34_VolumeSpike_v1"
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
    
    # Get 6h data for Williams %R calculation (based on previous 14 periods)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 15:
        return np.zeros(n)
    
    # Calculate 6h Williams %R: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = df_6h['high'].rolling(window=14, min_periods=14).max().values
    lowest_low = df_6h['low'].rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - df_6h['close'].values) / (highest_high - lowest_low) * -100
    # Replace division by zero with -50 (neutral)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align 6h Williams %R to 6h timeframe (no additional delay needed)
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 14-period average volume for confirmation
    vol_ma_14 = pd.Series(volume).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 34)  # Williams %R and 1d EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_14[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_williams_r = williams_r_aligned[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_vol_ma = vol_ma_14[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Williams %R crosses above -50 (mean reversion from oversold)
            if curr_williams_r > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -50 (mean reversion from overbought)
            if curr_williams_r < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.3x 14-period average (moderate filter)
            vol_confirmed = curr_volume > 1.3 * curr_vol_ma
            
            # Long when Williams %R crosses above -80 (oversold), price > 1d EMA34, volume confirmed
            if curr_williams_r > -80 and curr_close > curr_ema34_1d and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when Williams %R crosses below -20 (overbought), price < 1d EMA34, volume confirmed
            elif curr_williams_r < -20 and curr_close < curr_ema34_1d and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals
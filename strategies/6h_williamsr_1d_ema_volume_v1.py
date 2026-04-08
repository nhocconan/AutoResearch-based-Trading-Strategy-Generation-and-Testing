#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d trend filter and volume confirmation
# Uses Williams %R (14) on 6h for overbought/oversold signals:
# - Buy when Williams %R crosses above -80 (oversold) in 1d uptrend
# - Sell when Williams %R crosses below -20 (overbought) in 1d downtrend
# - 1d EMA50 filter ensures trades align with higher timeframe trend
# - Volume confirmation (current volume > 20-period average) avoids false signals
# Designed for low frequency (target: 15-35 trades/year) to minimize fee drag
# Williams %R is effective in ranging markets and captures reversals in trends

name = "6h_williamsr_1d_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams %R (14-period) on 6h data
    # Calculate on 6h data then align (though we're already on 6h, this ensures proper handling)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    diff = highest_high - lowest_low
    williams_r = np.where(diff != 0, -100 * (highest_high - close) / diff, -50)
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(williams_r[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Trend filter from 1d EMA
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Williams %R levels
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit when overbought or trend changes
            if overbought or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit when oversold or trend changes
            if oversold or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Entry conditions with trend and volume confirmation
            # Buy when coming out of oversold in uptrend
            if oversold and uptrend and vol_confirm and i > 0 and williams_r[i-1] >= -80:
                position = 1
                signals[i] = 0.25
            # Sell when coming out of overbought in downtrend
            elif overbought and downtrend and vol_confirm and i > 0 and williams_r[i-1] <= -20:
                position = -1
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 12h trend filter and volume confirmation
# - Williams %R(14) from 6h: oversold < -80 for long, overbought > -20 for short
# - 12h EMA(50) trend filter: only long when price > EMA50, short when price < EMA50
# - 6h volume confirmation: current volume > 1.5x 20-period volume SMA
# - Exit: Williams %R crosses above -50 (for long) or below -50 (for short)
# - Position sizing: 0.25 discrete level
# - Williams %R identifies extreme reversals, EMA filter ensures we trade with higher timeframe trend
# - Volume confirmation adds institutional participation signal
# - 6h timeframe targets 12-37 trades/year with strict entry conditions to minimize fee drag
# - Works in bull/bear: mean reversion at extremes works in all regimes, trend filter avoids counter-trend trades

name = "6h_12h_williamsr_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 6h Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    hh_ll = highest_high - lowest_low
    williams_r = np.where(hh_ll != 0, 
                          ((highest_high - close) / hh_ll) * -100, 
                          -50)  # default to neutral when invalid
    
    # Calculate 6h volume SMA(20) for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(volume_sma_20[i]) or 
            np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # Williams %R conditions
        williams_oversold = williams_r[i] < -80   # Oversold condition for long
        williams_overbought = williams_r[i] > -20 # Overbought condition for short
        williams_exit_long = williams_r[i] > -50  # Exit long when crosses above -50
        williams_exit_short = williams_r[i] < -50 # Exit short when crosses below -50
        
        # 12h EMA trend filter
        uptrend = close[i] > ema_50_12h_aligned[i]   # Price above 12h EMA50 = uptrend
        downtrend = close[i] < ema_50_12h_aligned[i] # Price below 12h EMA50 = downtrend
        
        # Entry conditions
        long_entry = williams_oversold and vol_confirm and uptrend
        short_entry = williams_overbought and vol_confirm and downtrend
        
        # Exit conditions
        long_exit = williams_exit_long
        short_exit = williams_exit_short
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if long_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if short_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals
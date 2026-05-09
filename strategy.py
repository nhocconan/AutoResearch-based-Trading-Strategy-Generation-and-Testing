#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R reversal with 1d trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. In strong trends, pullbacks to
# extreme %R levels offer high-probability entries. Long when %R crosses above -80 from oversold
# with 1d uptrend and volume spike; short when crosses below -20 from overbought with 1d downtrend.
# Uses Williams %R(14) on 12h chart, filtered by 1d EMA(34) trend and volume > 1.5x 20-period EMA.
# Designed for fewer trades (target: 15-30/year) to avoid fee drag in ranging markets.
name = "12h_WilliamsR_Reversal_1dEMA_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams %R calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R on 12h: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_12h['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_12h['low'].values).rolling(window=14, min_periods=14).min().values
    close_12h = df_12h['close'].values
    williams_r = (highest_high - close_12h) / (highest_high - lowest_low) * -100
    
    # Align Williams %R to 12h timeframe (no additional delay needed as it's based on current bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 14  # Need 14 periods for Williams %R
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R values
        wr = williams_r_aligned[i]
        wr_prev = williams_r_aligned[i-1] if i > 0 else wr
        
        price = close[i]
        
        if position == 0:
            # Enter long: Williams %R crosses above -80 from oversold + 1d uptrend + volume spike
            if (wr > -80 and wr_prev <= -80 and price > ema_34_1d_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R crosses below -20 from overbought + 1d downtrend + volume spike
            elif (wr < -20 and wr_prev >= -20 and price < ema_34_1d_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R rises above -20 (overbought) or trend reverses
            if wr > -20 or price < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R falls below -80 (oversold) or trend reverses
            if wr < -80 or price > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
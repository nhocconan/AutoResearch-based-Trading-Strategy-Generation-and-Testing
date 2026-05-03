#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when: price breaks above 12h Camarilla R3 level AND close > 1d EMA34 AND volume > 2.0x 24-bar average
# Short when: price breaks below 12h Camarilla S3 level AND close < 1d EMA34 AND volume > 2.0x 24-bar average
# Exit via ATR(24) trailing stop: long exit when price < highest_high_since_entry - 2.5 * ATR
#                      short exit when price > lowest_low_since_entry + 2.5 * ATR
# Uses 12h Camarilla for structure (proven edge from top performers), 1d EMA34 for HTF trend alignment, volume spike for confirmation
# Discrete sizing 0.25 balances return and fee drag. Target: 50-150 total trades over 4 years = 12-37/year.

name = "12h_Camarilla_R3_S3_1dEMA34_VolumeSpike_ATRStop_v1"
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
    
    # Calculate 12h Camarilla pivots (based on previous 12h bar)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Previous 12h bar for Camarilla calculation (index -2 for current forming bar)
    ph = high_12h[-2]  # previous 12h high
    pl = low_12h[-2]   # previous 12h low
    pc = close_12h[-2] # previous 12h close
    
    # Camarilla levels
    R3 = pc + (ph - pl) * 1.1 / 4
    S3 = pc - (ph - pl) * 1.1 / 4
    
    # Calculate 1d EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume spike confirmation (24-bar average)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (vol_ma_24 * 2.0)
    
    # Calculate ATR(24) for trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_24 = pd.Series(tr).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high = 0.0
    lowest_low = 0.0
    
    for i in range(100, n):
        # Entry conditions
        long_entry = (
            close[i] > R3 and  # price above 12h Camarilla R3
            close[i] > ema_34_1d_aligned[i] and  # above 1d EMA34 trend
            volume_spike[i]  # volume confirmation
        )
        
        short_entry = (
            close[i] < S3 and  # price below 12h Camarilla S3
            close[i] < ema_34_1d_aligned[i] and  # below 1d EMA34 trend
            volume_spike[i]  # volume confirmation
        )
        
        # Exit conditions (trailing stop)
        long_exit = False
        short_exit = False
        
        if position == 1:  # long position
            highest_high = max(highest_high, high[i])
            long_exit = close[i] < (highest_high - 2.5 * atr_24[i])
        elif position == -1:  # short position
            lowest_low = min(lowest_low, low[i])
            short_exit = close[i] > (lowest_low + 2.5 * atr_24[i])
        
        # Generate signal
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
                highest_high = high[i]
            elif short_entry:
                signals[i] = -0.25
                position = -1
                lowest_low = low[i]
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
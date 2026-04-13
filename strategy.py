#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1d RSI filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. In strong trends, it can remain
# extreme for extended periods. Combined with 1d RSI trend filter (avoid counter-trend trades)
# and volume spikes, it captures high-probability mean-reversion entries in the direction
# of the higher timeframe trend. Designed to work in both bull and bear markets by
# aligning with the 1d trend. Target: 20-50 trades per year (80-200 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # RSI(14) for 1d trend filter
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])  # first average of first 14 gains
    avg_loss[13] = np.mean(loss[1:14])  # first average of first 14 losses
    
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d[:13] = np.nan  # first 13 values are not valid
    
    # Align 1d RSI to 4h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Williams %R on 4h timeframe (14-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    williams_r = np.full(n, np.nan)
    
    for i in range(13, n):  # need 14 periods for %R calculation
        highest_high[i] = np.max(high[i-13:i+1])
        lowest_low[i] = np.min(low[i-13:i+1])
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            williams_r[i] = -50  # avoid division by zero
    
    # Average volume (20-period = 10 hours) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(williams_r[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r[i]
        rsi = rsi_1d_aligned[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirm = vol > 2.0 * avg_vol
        
        if position == 0:
            # Long: Oversold (Williams %R < -80) + 1d RSI > 50 (uptrend filter) + volume confirmation
            if (wr < -80 and rsi > 50 and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Overbought (Williams %R > -20) + 1d RSI < 50 (downtrend filter) + volume confirmation
            elif (wr > -20 and rsi < 50 and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns above -50 or 1d RSI < 40
            if (wr > -50 or rsi < 40):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns below -50 or 1d RSI > 60
            if (wr < -50 or rsi > 60):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_WilliamsR_RSI_Volume"
timeframe = "4h"
leverage = 1.0
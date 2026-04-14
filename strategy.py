#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1d RSI mean reversion and 1w volume confirmation
# RSI(14) on daily timeframe identifies overbought/oversold conditions
# Volume spike on weekly timeframe confirms institutional participation
# Combines mean reversion with volume confirmation to filter false signals
# Works in both bull and bear markets as it captures reversals at extremes
# Uses strict entry conditions to limit trades (<50/year) and avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for RSI
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d RSI (14)
    rsi_length = 14
    delta = np.diff(df_1d['close'].values, prepend=df_1d['close'].values[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (alpha = 1/length)
    alpha = 1.0 / rsi_length
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[rsi_length] = np.mean(gain[1:rsi_length+1])
    avg_loss[rsi_length] = np.mean(loss[1:rsi_length+1])
    
    for i in range(rsi_length+1, len(gain)):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i-1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i-1]
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:rsi_length] = np.nan  # Not enough data
    
    # Align RSI to 4h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Load 1w data ONCE for volume average
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w average volume (20 periods)
    vol_ma_length = 20
    vol_ma = pd.Series(df_1w['volume']).rolling(window=vol_ma_length, min_periods=vol_ma_length).mean().values
    
    # Align volume MA to 4h timeframe
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(100, rsi_length + 10, vol_ma_length)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Volume confirmation: current volume > 1.5x weekly average
        vol_confirm = vol > 1.5 * vol_ma_aligned[i]
        
        # RSI extremes: oversold (<30) or overbought (>70)
        rsi_oversold = rsi_aligned[i] < 30
        rsi_overbought = rsi_aligned[i] > 70
        
        if position == 0:
            # Enter long: RSI oversold + volume confirmation
            if rsi_oversold and vol_confirm:
                position = 1
                signals[i] = position_size
            # Enter short: RSI overbought + volume confirmation
            elif rsi_overbought and vol_confirm:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral (40-60) OR volume drops
            rsi_neutral = 40 <= rsi_aligned[i] <= 60
            vol_drop = vol < vol_ma_aligned[i]  # Volume below average
            
            if rsi_neutral or vol_drop:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI returns to neutral OR volume drops
            if rsi_neutral or vol_drop:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1dRSI_1wVolume_MeanReversion_v1"
timeframe = "4h"
leverage = 1.0
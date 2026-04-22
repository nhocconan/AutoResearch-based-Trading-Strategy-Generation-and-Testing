# 12h_Camarilla_Pivot_Reversal_1dTrend_Volume
# Hypothesis: 12-hour Camarilla pivot reversal with 1-day trend and volume confirmation
# Long at S1 when 1d EMA34 rising with volume spike, short at R1 when 1d EMA34 falling with volume spike
# Exit when price crosses Camarilla pivot or 1d EMA34 reverses
# Designed for low trade frequency (12-37/year) with multiple confirmations
# Works in both bull and bear markets by following 1d trend while using 12h for entry timing

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels for 12h timeframe
    high_12h = get_htf_data(prices, '12h')['high'].values
    low_12h = get_htf_data(prices, '12h')['low'].values
    close_12h = get_htf_data(prices, '12h')['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = Close + ((High-Low) * 1.1/2)
    # R3 = Close + ((High-Low) * 1.1/4)
    # R2 = Close + ((High-Low) * 1.1/6)
    # R1 = Close + ((High-Low) * 1.1/12)
    # PP = (High + Low + Close)/3
    # S1 = Close - ((High-Low) * 1.1/12)
    # S2 = Close - ((High-Low) * 1.1/6)
    # S3 = Close - ((High-Low) * 1.1/4)
    # S4 = Close - ((High-Low) * 1.1/2)
    
    # We need previous 12h bar's data for current 12h bar's levels
    # Since we're working with 12h data, we shift by 1 to get previous bar
    if len(close_12h) < 2:
        return np.zeros(n)
    
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan  # First value invalid
    
    # Calculate Camarilla levels for each 12h bar
    camarilla_r1 = prev_close + ((prev_high - prev_low) * 1.1 / 12)
    camarilla_s1 = prev_close - ((prev_high - prev_low) * 1.1 / 12)
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    
    # Align Camarilla levels to 12h timeframe first
    r1_12h = align_htf_to_ltf(prices, get_htf_data(prices, '12h'), camarilla_r1)
    s1_12h = align_htf_to_ltf(prices, get_htf_data(prices, '12h'), camarilla_s1)
    pp_12h = align_htf_to_ltf(prices, get_htf_data(prices, '12h'), camarilla_pp)
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 34-period EMA on 1d close for trend
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average (on 12h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or np.isnan(pp_12h[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: price at S1, 1d EMA34 rising, volume spike
            if (close[i] <= s1_12h[i] * 1.001 and  # Allow small buffer for touching
                ema34_1d_aligned[i] > ema34_1d_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: price at R1, 1d EMA34 falling, volume spike
            elif (close[i] >= r1_12h[i] * 0.999 and   # Allow small buffer for touching
                  ema34_1d_aligned[i] < ema34_1d_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses pivot or 1d EMA34 reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses above pivot or 1d EMA34 turns down
                if close[i] >= pp_12h[i] or ema34_1d_aligned[i] < ema34_1d_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses below pivot or 1d EMA34 turns up
                if close[i] <= pp_12h[i] or ema34_1d_aligned[i] > ema34_1d_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Camarilla_Pivot_Reversal_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0
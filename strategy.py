#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot level reversal with 1d EMA34 trend filter and volume spike confirmation.
# Camarilla levels identify key support/resistance where price often reverses. Combined with 1d EMA34 for trend direction
# and volume spikes (>1.5x 20-period average) to confirm institutional interest, this captures reversals at key levels.
# Designed for low trade frequency (~20-35/year) to minimize fee decay. Works in both bull and bear markets by trading
# reversals at institutional levels with trend alignment.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data for Camarilla calculation and EMA34 (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (based on previous day's range)
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # R2 = C + ((H-L) * 1.1/6)
    # R1 = C + ((H-L) * 1.1/12)
    # PP = (H+L+C)/3
    # S1 = C - ((H-L) * 1.1/12)
    # S2 = C - ((H-L) * 1.1/6)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    
    # Use previous day's data (shift by 1 to avoid look-ahead)
    if len(high_1d) < 2:
        return np.zeros(n)
    
    prev_high = np.roll(high_1d, 1)[1:]  # Shift right, first element will be overwritten
    prev_low = np.roll(low_1d, 1)[1:]
    prev_close = np.roll(close_1d, 1)[1:]
    
    # Pad the first element with NaN (will be handled by alignment)
    prev_high = np.concatenate([[np.nan], prev_high[:-1]])
    prev_low = np.concatenate([[np.nan], prev_low[:-1]])
    prev_close = np.concatenate([[np.nan], prev_close[:-1]])
    
    # Calculate pivot levels using previous day's data
    hl_range = prev_high - prev_low
    camarilla_pp = (prev_high + prev_low + prev_close) / 3.0
    camarilla_r1 = camarilla_pp + (hl_range * 1.1 / 12)
    camarilla_r2 = camarilla_pp + (hl_range * 1.1 / 6)
    camarilla_r3 = camarilla_pp + (hl_range * 1.1 / 4)
    camarilla_r4 = camarilla_pp + (hl_range * 1.1 / 2)
    camarilla_s1 = camarilla_pp - (hl_range * 1.1 / 12)
    camarilla_s2 = camarilla_pp - (hl_range * 1.1 / 6)
    camarilla_s3 = camarilla_pp - (hl_range * 1.1 / 4)
    camarilla_s4 = camarilla_pp - (hl_range * 1.1 / 2)
    
    # Calculate 34-period EMA on 1d close for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 4h timeframe (waits for 1d bar to close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_r2_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_s2_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        r1 = camarilla_r1_aligned[i]
        r2 = camarilla_r2_aligned[i]
        s1 = camarilla_s1_aligned[i]
        s2 = camarilla_s2_aligned[i]
        ema_val = ema_34_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average (balanced filter)
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: price near S1/S2 + uptrend + volume spike
            near_support = (price <= s1 * 1.002 and price >= s2 * 0.998) or \
                           (price <= s2 * 1.002 and price >= s1 * 0.998)
            if near_support and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price near R1/R2 + downtrend + volume spike
            elif near_resistance := ((price <= r1 * 1.002 and price >= r2 * 0.998) or \
                           (price <= r2 * 1.002 and price >= r1 * 0.998)):
                if price < ema_val and vol_spike:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price reaches R1 or breaks below S2 or trend breaks
                if price >= r1 * 0.998 or price <= s2 * 1.002 or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price reaches S1 or breaks above R2 or trend breaks
                if price <= s1 * 1.002 or price >= r2 * 0.998 or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Reversal_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0
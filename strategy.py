#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R (14) extreme readings with volume confirmation and 4h Supertrend trend filter.
# Enter long when 1d Williams %R < -80 (oversold) with volume spike and price above 4h Supertrend.
# Enter short when 1d Williams %R > -20 (overbought) with volume spike and price below 4h Supertrend.
# Uses discrete position sizing (0.25) to balance return and drawdown. Target: 20-50 trades/year.
# Williams %R provides mean reversion signals from higher timeframe, volume confirms reversal strength, Supertrend filters intermediate trend.
# Works in bull (bounces from oversold) and bear (rejections from overbought) markets.

name = "4h_WilliamsR14_4hSupertrend_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    n_1d = len(high_1d)
    williams_r = np.full(n_1d, np.nan)
    
    for i in range(13, n_1d):  # Start from index 13 for 14-period lookback
        highest_high = np.max(high_1d[i-13:i+1])
        lowest_low = np.min(low_1d[i-13:i+1])
        if highest_high != lowest_low:
            williams_r[i] = (highest_high - close_1d[i]) / (highest_high - lowest_low) * -100
        else:
            williams_r[i] = -50  # Neutral when range is zero
    
    # Forward fill Williams %R
    williams_r = pd.Series(williams_r).ffill().values
    
    # Align 1d Williams %R to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Get 4h data for Supertrend trend filter
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 10:
        return np.zeros(n)
    
    # Calculate 4h Supertrend (10, 3.0)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ATR
    tr1 = np.abs(high_4h[1:] - low_4h[1:])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    atr_period = 10
    atr = np.full(len(tr), np.nan)
    for i in range(atr_period, len(tr)):
        if i == atr_period:
            atr[i] = np.nanmean(tr[i-atr_period+1:i+1])
        else:
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Calculate Supertrend
    upper_band = np.full(len(close_4h), np.nan)
    lower_band = np.full(len(close_4h), np.nan)
    supertrend = np.full(len(close_4h), np.nan)
    uptrend = np.full(len(close_4h), True)
    
    multiplier = 3.0
    
    for i in range(1, len(close_4h)):
        if np.isnan(atr[i]) or np.isnan(atr[i-1]):
            upper_band[i] = upper_band[i-1]
            lower_band[i] = lower_band[i-1]
            supertrend[i] = supertrend[i-1]
            uptrend[i] = uptrend[i-1]
            continue
        
        upper_band[i] = (high_4h[i] + low_4h[i]) / 2 + multiplier * atr[i]
        lower_band[i] = (high_4h[i] + low_4h[i]) / 2 - multiplier * atr[i]
        
        if i == 1:
            upper_band[i] = upper_band[i]
            lower_band[i] = lower_band[i]
        else:
            if close_4h[i-1] > upper_band[i-1]:
                upper_band[i] = upper_band[i]
            else:
                upper_band[i] = min(upper_band[i], upper_band[i-1])
            
            if close_4h[i-1] < lower_band[i-1]:
                lower_band[i] = lower_band[i]
            else:
                lower_band[i] = max(lower_band[i], lower_band[i-1])
        
        if i == 1:
            supertrend[i] = upper_band[i]
            uptrend[i] = True
        else:
            if supertrend[i-1] == upper_band[i-1]:
                if close_4h[i] <= upper_band[i]:
                    supertrend[i] = upper_band[i]
                    uptrend[i] = True
                else:
                    supertrend[i] = lower_band[i]
                    uptrend[i] = False
            else:
                if close_4h[i] >= lower_band[i]:
                    supertrend[i] = lower_band[i]
                    uptrend[i] = True
                else:
                    supertrend[i] = upper_band[i]
                    uptrend[i] = False
    
    # Forward fill Supertrend
    supertrend = pd.Series(supertrend).ffill().values
    
    # Align Supertrend to 4h timeframe (already aligned since primary is 4h, but keep for consistency)
    supertrend_aligned = align_htf_to_ltf(prices, df_4h, supertrend)
    
    # Calculate 4h volume spike: >2.0x 20-bar average volume (equivalent to ~1.33d)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(supertrend_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 4h Supertrend
        above_supertrend = close[i] > supertrend_aligned[i]
        below_supertrend = close[i] < supertrend_aligned[i]
        
        # Williams %R extreme conditions with volume confirmation
        long_signal = williams_r_aligned[i] < -80 and volume_spike[i]  # Oversold
        short_signal = williams_r_aligned[i] > -20 and volume_spike[i]  # Overbought
        
        # Exit conditions: opposite extreme or trend reversal
        long_exit = williams_r_aligned[i] > -20 or below_supertrend  # Overbought or trend down
        short_exit = williams_r_aligned[i] < -80 or above_supertrend  # Oversold or trend up
        
        # Handle entries and exits
        if long_signal and above_supertrend and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_signal and below_supertrend and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals
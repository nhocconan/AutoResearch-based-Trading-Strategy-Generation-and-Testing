#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h/1d Camarilla pivot breakout + volume spike + session filter
    # Uses 4h for structure (Camarilla levels), 1d for trend filter (close vs SMA50)
    # 1h for entry timing with volume confirmation (volume > 1.5x 20-period average)
    # Session filter: 08-20 UTC to avoid low-volume Asian session noise
    # Discrete sizing: 0.20 to control drawdown and minimize fee churn
    # Target: 60-150 trades over 4 years (15-37/year) for 1h timeframe
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (using previous 4h bar's OHLC)
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    open_4h = df_4h['open'].values
    
    # True range for Camarilla: (high - low) of previous 4h bar
    tr_4h = high_4h - low_4h
    
    # Camarilla levels (based on previous 4h bar)
    # L3 = close + 1.1*(high-low)/12, H3 = close - 1.1*(high-low)/12
    # L4 = close + 1.1*(high-low)/6, H4 = close - 1.1*(high-low)/6
    camarilla_l3 = close_4h + 1.1 * tr_4h / 12
    camarilla_h3 = close_4h - 1.1 * tr_4h / 12
    camarilla_l4 = close_4h + 1.1 * tr_4h / 6
    camarilla_h4 = close_4h - 1.1 * tr_4h / 6
    
    # Align Camarilla levels to 1h timeframe (wait for 4h bar close)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l4)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h4)
    
    # 1d trend filter: close vs SMA50
    close_1d = df_1d['close'].values
    sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% position size
    
    for i in range(100, n):
        # Skip if not in trading session or data not ready
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        if (np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(sma50_1d_aligned[i]) or np.isnan(close_1d_aligned[i]) or
            np.isnan(vol_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Long breakout: price breaks above H3 or H4 with volume spike in bull regime
        long_breakout = ((close[i] > camarilla_h3_aligned[i]) or (close[i] > camarilla_h4_aligned[i])) and \
                        volume_spike[i] and \
                        (close_1d_aligned[i] > sma50_1d_aligned[i])
        
        # Short breakout: price breaks below L3 or L4 with volume spike in bear regime
        short_breakout = ((close[i] < camarilla_l3_aligned[i]) or (close[i] < camarilla_l4_aligned[i])) and \
                         volume_spike[i] and \
                         (close_1d_aligned[i] < sma50_1d_aligned[i])
        
        # Exit conditions: opposite breakout or loss of volume/momentum
        exit_long = ((close[i] < camarilla_l3_aligned[i]) or 
                     (not volume_spike[i]) or
                     (close_1d_aligned[i] < sma50_1d_aligned[i]))
        exit_short = ((close[i] > camarilla_h3_aligned[i]) or 
                      (not volume_spike[i]) or
                      (close_1d_aligned[i] > sma50_1d_aligned[i]))
        
        # Entry logic
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit logic
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_camarilla_breakout_volume_v1"
timeframe = "1h"
leverage = 1.0
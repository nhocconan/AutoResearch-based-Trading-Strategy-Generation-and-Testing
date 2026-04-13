#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation
    # Works in both bull and bear: Camarilla captures intraday reversals/breakouts,
    # 4h trend filters false signals, volume confirms momentum.
    # Target: 15-35 trades/year (~60-140 over 4 years) to minimize fee drag.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1h data for Camarilla pivot calculation (using prior day's OHLC)
    df_1h = prices.copy()
    
    # Calculate prior day's OHLC for Camarilla levels
    # We'll use daily OHLC from 1d timeframe for accuracy
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_h4 = np.zeros(len(df_1d))
    camarilla_h3 = np.zeros(len(df_1d))
    camarilla_h2 = np.zeros(len(df_1d))
    camarilla_h1 = np.zeros(len(df_1d))
    camarilla_l1 = np.zeros(len(df_1d))
    camarilla_l2 = np.zeros(len(df_1d))
    camarilla_l3 = np.zeros(len(df_1d))
    camarilla_l4 = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        if i == 0:
            # For first day, use available data
            camarilla_h4[i] = high_1d[i]
            camarilla_h3[i] = high_1d[i]
            camarilla_h2[i] = high_1d[i]
            camarilla_h1[i] = high_1d[i]
            camarilla_l1[i] = low_1d[i]
            camarilla_l2[i] = low_1d[i]
            camarilla_l3[i] = low_1d[i]
            camarilla_l4[i] = low_1d[i]
        else:
            # Camarilla formula using prior day's OHLC
            close_prev = close_1d[i-1]
            high_prev = high_1d[i-1]
            low_prev = low_1d[i-1]
            range_prev = high_prev - low_prev
            
            camarilla_h4[i] = close_prev + range_prev * 1.1 / 2
            camarilla_h3[i] = close_prev + range_prev * 1.1 / 4
            camarilla_h2[i] = close_prev + range_prev * 1.1 / 6
            camarilla_h1[i] = close_prev + range_prev * 1.1 / 12
            camarilla_l1[i] = close_prev - range_prev * 1.1 / 12
            camarilla_l2[i] = close_prev - range_prev * 1.1 / 6
            camarilla_l3[i] = close_prev - range_prev * 1.1 / 4
            camarilla_l4[i] = close_prev - range_prev * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get volume confirmation from 1h (current volume > 1.5x 20-period average)
    vol_avg_20_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(vol_avg_20_1h[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        if hour < 8 or hour > 20:
            # Outside session: flatten position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_avg_20_1h[i]
        
        # Trend direction from 4h EMA(50)
        trend_up = close[i] > ema_50_4h_aligned[i]
        trend_down = close[i] < ema_50_4h_aligned[i]
        
        # Entry conditions: Camarilla breakout + trend + volume
        # Long: break above H3 with uptrend
        enter_long = (close[i] > camarilla_h3_aligned[i]) and trend_up and volume_confirmed
        # Short: break below L3 with downtrend
        enter_short = (close[i] < camarilla_l3_aligned[i]) and trend_down and volume_confirmed
        
        # Exit conditions: reverse signal or Camarilla level touch
        exit_long = position == 1 and (close[i] < camarilla_h1_aligned[i] or not trend_up)
        exit_short = position == -1 and (close[i] > camarilla_l1_aligned[i] or not trend_down)
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_camarilla_4htrend_volume_session_v1"
timeframe = "1h"
leverage = 1.0
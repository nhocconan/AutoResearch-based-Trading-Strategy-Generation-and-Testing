#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_camarilla_breakout_v30
# Camarilla pivot breakout on 4h timeframe with 1d trend filter (EMA21).
# Uses Camarilla levels (H4/L4) as dynamic support/resistance.
# Breakout confirmed by volume spike (>1.5x 20-period average).
# Works in both bull and bear markets by trading breakouts in direction of 1d trend.
# Low-moderate trade frequency expected (~25-35/year) due to specific breakout conditions.
name = "4h_1d_camarilla_breakout_v30"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (EMA21)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate 1d EMA21 for trend filter
    close_1d = df_1d['close'].values
    ema_21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 1d EMA21 to 4h timeframe
    ema_21_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # Calculate Camarilla levels for each 4h bar using prior 1d OHLC
    # We need prior 1d OHLC, so we'll shift the 1d data by 1 bar
    # This ensures we only use completed 1d bars for Camarilla calculation
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get prior 1d OHLC (shifted by 1 to avoid look-ahead)
    prior_high_1d = df_1d['high'].shift(1).values
    prior_low_1d = df_1d['low'].shift(1).values
    prior_close_1d = df_1d['close'].shift(1).values
    
    # Align prior 1d OHLC to 4h timeframe
    prior_high_1d_aligned = align_htf_to_ltf(prices, df_1d, prior_high_1d)
    prior_low_1d_aligned = align_htf_to_ltf(prices, df_1d, prior_low_1d)
    prior_close_1d_aligned = align_htf_to_ltf(prices, df_1d, prior_close_1d)
    
    # Calculate Camarilla levels: H4, L4
    # H4 = Close + 1.1 * (High - Low) / 2
    # L4 = Close - 1.1 * (High - Low) / 2
    camarilla_h4 = prior_close_1d_aligned + 1.1 * (prior_high_1d_aligned - prior_low_1d_aligned) / 2
    camarilla_l4 = prior_close_1d_aligned - 1.1 * (prior_high_1d_aligned - prior_low_1d_aligned) / 2
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after volume MA warmup
        # Skip if EMA not ready
        if np.isnan(ema_21_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
            
        # Breakout conditions
        bullish_breakout = (close[i] > camarilla_h4[i]) and volume_filter[i]
        bearish_breakout = (close[i] < camarilla_l4[i]) and volume_filter[i]
        
        # Trend filter: only trade in direction of 1d trend
        bullish_signal = bullish_breakout and (close[i] > ema_21_aligned[i])
        bearish_signal = bearish_breakout and (close[i] < ema_21_aligned[i])
        
        # Exit on opposite breakout
        exit_long = bearish_breakout
        exit_short = bullish_breakout
        
        if bullish_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals
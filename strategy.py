#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation.
# Uses proven Camarilla pivot structure with weekly trend filter for regime adaptation.
# Long when price breaks above Camarilla R3 with volume and price > 1w EMA34 (uptrend).
# Short when price breaks below Camarilla S3 with volume and price < 1w EMA34 (downtrend).
# Volume spike (>2.0x 24-bar average) confirms breakout strength.
# Position size 0.25 balances return and drawdown. Discrete levels minimize fee churn.
# Works in both bull and bear via 1w EMA34 trend filter and strict entry conditions.

name = "12h_Camarilla_R3S3_1wEMA34_Trend_VolumeSpike_v1"
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
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data for Camarilla pivot levels (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3 and S3 levels: R3 = close + 1.1*(high-low)/4, S3 = close - 1.1*(high-low)/4
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 4
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 12h volume spike: >2.0x 24-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure sufficient history for EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1w EMA34 direction (price above/below EMA34)
        price_above_ema = close[i] > ema_34_1w_aligned[i]
        price_below_ema = close[i] < ema_34_1w_aligned[i]
        
        # Camarilla breakout conditions with volume confirmation
        long_breakout = close[i] > camarilla_r3_aligned[i] and volume_spike[i]
        short_breakout = close[i] < camarilla_s3_aligned[i] and volume_spike[i]
        
        # Exit conditions: opposite Camarilla level or trend reversal
        long_exit = close[i] < camarilla_s3_aligned[i] or close[i] < ema_34_1w_aligned[i]
        short_exit = close[i] > camarilla_r3_aligned[i] or close[i] > ema_34_1w_aligned[i]
        
        # Handle entries and exits
        if long_breakout and price_above_ema and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and price_below_ema and position >= 0:
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
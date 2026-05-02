#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R mean reversion with 1w EMA34 trend filter and volume spike confirmation
# Williams %R identifies overbought/oversold conditions. Mean reversion from extreme levels
# (>80 for short, <20 for long) with 1w EMA34 trend alignment captures reversals in both bull and bear markets.
# Volume spike (2.0x 20-period average) filters false signals. Discrete sizing 0.25 targets ~50-100 trades over 4 years.
# Timeframe: 1d (slower timeframe minimizes fee drag, improves test generalization in bear markets).

name = "1d_WilliamsR_MeanReversion_1wEMA34_Trend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for Williams %R and EMA34
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate Williams %R on 1w: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1w['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1w['low']).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - df_1w['close'].values) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 1d (wait for completed 1w bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    
    # Calculate EMA(34) on 1w for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation (2.0x 20-period average) on 1d
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Williams %R and EMA calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R < -80 (oversold) with 1w uptrend (close > EMA34)
            long_signal = williams_r_aligned[i] < -80
            # Short entry: Williams %R > -20 (overbought) with 1w downtrend (close < EMA34)
            short_signal = williams_r_aligned[i] > -20
            
            # 1w EMA34 trend filter: close above/below EMA indicates trend direction
            ema_trend_up = close[i] > ema_34_1w_aligned[i]
            ema_trend_down = close[i] < ema_34_1w_aligned[i]
            
            if long_signal and ema_trend_up and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            elif short_signal and ema_trend_down and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R > -20 (overbought) or trend reversal (close < EMA34)
            if williams_r_aligned[i] > -20 or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R < -80 (oversold) or trend reversal (close > EMA34)
            if williams_r_aligned[i] < -80 or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
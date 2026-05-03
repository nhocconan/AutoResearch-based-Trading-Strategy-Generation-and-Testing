#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R reversal with 12h EMA50 trend filter and volume confirmation.
# Long when Williams %R crosses above -80 from below in 12h uptrend with volume spike.
# Short when Williams %R crosses below -20 from above in 12h downtrend with volume spike.
# Uses ATR-based stoploss (signal→0 when price moves against position by 2.5*ATR).
# Designed for 4h timeframe to balance trade frequency and fee drag. Target: 75-200 total trades over 4 years.
# Williams %R identifies overbought/oversold conditions, 12h EMA50 ensures higher timeframe alignment,
# Volume spike confirms institutional interest. Works in both bull and bear markets by only trading
# with the 12h trend, avoiding counter-trend whipsaws.

name = "4h_WilliamsR_12hEMA50_VolumeSpike_ATR"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR for stoploss (using primary timeframe)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 4h data for Williams %R calculation
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Calculate 4h Williams %R (14-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_4h) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # Avoid division by zero
    williams_r_aligned = align_htf_to_ltf(prices, df_4h, williams_r)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike detection (20-period volume MA on primary timeframe)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # Track entry price for ATR-based stoploss
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        vol_spike = volume_spike[i]
        wr = williams_r_aligned[i]
        wr_prev = williams_r_aligned[i-1] if i > 0 else -50
        trend_up = close_val > ema_50_12h_aligned[i]   # 12h uptrend
        trend_down = close_val < ema_50_12h_aligned[i]  # 12h downtrend
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below AND 12h uptrend AND volume spike
            if wr > -80 and wr_prev <= -80 and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: Williams %R crosses below -20 from above AND 12h downtrend AND volume spike
            elif wr < -20 and wr_prev >= -20 and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            # Stoploss: price moves against position by 2.5*ATR
            if close_val < entry_price - 2.5 * atr[i]:
                exit_signal = True
            # Exit: Williams %R crosses above -20 (overbought)
            elif wr > -20 and wr_prev <= -20:
                exit_signal = True
            # Exit: 12h trend changes to downtrend
            elif not trend_up:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions for short
            exit_signal = False
            # Stoploss: price moves against position by 2.5*ATR
            if close_val > entry_price + 2.5 * atr[i]:
                exit_signal = True
            # Exit: Williams %R crosses below -80 (oversold)
            elif wr < -80 and wr_prev >= -80:
                exit_signal = True
            # Exit: 12h trend changes to uptrend
            elif not trend_down:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
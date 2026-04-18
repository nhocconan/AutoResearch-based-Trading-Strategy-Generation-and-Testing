#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1-week Stochastic RSI filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. Stochastic RSI on weekly timeframe
# filters for momentum alignment. Volume confirmation ensures breakout conviction.
# Designed for low trade frequency (15-30/year) to minimize fee drag in 6h timeframe.
# Works in bull markets (oversold bounce with bullish weekly momentum) and bear markets 
# (overbought rejection with bearish weekly momentum).
name = "6h_WilliamsR_StochRSI_Weekly_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams %R calculation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    # Get weekly data for Stochastic RSI filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Williams %R (14-period) using previous day's data
    highest_high_14 = df_1d['high'].rolling(window=14, min_periods=14).max().shift(1).values
    lowest_low_14 = df_1d['low'].rolling(window=14, min_periods=14).min().shift(1).values
    williams_r = -100 * (highest_high_14 - df_1d['close'].shift(1).values) / (highest_high_14 - lowest_low_14)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)  # avoid div by zero
    
    # Calculate Stochastic RSI on weekly timeframe
    # RSI(14) on weekly close
    delta = df_1w['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Stochastic RSI: (RSI - min(RSI)) / (max(RSI) - min(RSI)) over 14 periods
    rsi_min_14 = pd.Series(rsi).rolling(window=14, min_periods=14).min().values
    rsi_max_14 = pd.Series(rsi).rolling(window=14, min_periods=14).max().values
    stoch_rsi = (rsi - rsi_min_14) / (rsi_max_14 - rsi_min_14)
    stoch_rsi = np.where((rsi_max_14 - rsi_min_14) == 0, 0.5, stoch_rsi)  # avoid div by zero
    
    stoch_rsi_smoothed = pd.Series(stoch_rsi).rolling(window=3, min_periods=3).mean().values  # smooth with 3-period MA
    
    # Align indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    stoch_rsi_aligned = align_htf_to_ltf(prices, df_1w, stoch_rsi_smoothed)
    
    # Calculate 24-period average volume for confirmation (4 days of 6h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or np.isnan(stoch_rsi_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_24[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND Stochastic RSI rising from oversold AND volume
            wr_oversold = williams_r_aligned[i] < -80
            stoch_rising = stoch_rsi_aligned[i] > stoch_rsi_aligned[i-1] if i > 0 else False
            stoch_oversold = stoch_rsi_aligned[i] < 0.2  # Stochastic RSI oversold
            
            if vol_confirm and wr_oversold and stoch_rising and stoch_oversold:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND Stochastic RSI falling from overbought AND volume
            elif (vol_confirm and 
                  williams_r_aligned[i] > -20 and 
                  stoch_rsi_aligned[i] < stoch_rsi_aligned[i-1] if i > 0 else False and  # stoch falling
                  stoch_rsi_aligned[i] > 0.8):  # Stochastic RSI overbought
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R rises above -50 OR Stochastic RSI becomes overbought
            wr_exit = williams_r_aligned[i] > -50
            stoch_overbought = stoch_rsi_aligned[i] > 0.8
            
            if wr_exit or stoch_overbought:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R falls below -50 OR Stochastic RSI becomes oversold
            wr_exit = williams_r_aligned[i] < -50
            stoch_oversold = stoch_rsi_aligned[i] < 0.2
            
            if wr_exit or stoch_oversold:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band Breakout + 1w RSI Momentum + Volume Confirmation
# Long when price breaks above upper Bollinger Band(20,2) and 1w RSI > 50 and volume > 1.5x 20-day average
# Short when price breaks below lower Bollinger Band(20,2) and 1w RSI < 50 and volume > 1.5x 20-day average
# Exit when price crosses Bollinger middle band (20-day SMA)
# Bollinger Bands capture volatility expansion/contraction
# 1w RSI ensures alignment with higher-timeframe momentum
# Volume confirmation avoids false breakouts
# Target: 15-25 trades/year by requiring Bollinger breakout + RSI filter + volume spike

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d Bollinger Bands(20,2)
    close_1d = df_1d['close'].values
    bb_period = 20
    bb_std = 2
    
    sma = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std().values
    
    upper_band = sma + (bb_std * std)
    lower_band = sma - (bb_std * std)
    middle_band = sma
    
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    middle_band_aligned = align_htf_to_ltf(prices, df_1d, middle_band)
    
    # Calculate 1w RSI(14)
    close_1w = df_1w['close'].values
    rsi_period = 14
    
    delta = np.diff(close_1d)  # Use 1d close for RSI calculation on daily data
    # Need to align delta to 1d array length
    delta_full = np.zeros_like(close_1d)
    delta_full[1:] = delta
    
    gain = np.where(delta_full > 0, delta_full, 0)
    loss = np.where(delta_full < 0, -delta_full, 0)
    
    # Wilder's smoothing for RSI
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    
    avg_gain[rsi_period] = np.mean(gain[:rsi_period+1])
    avg_loss[rsi_period] = np.mean(loss[:rsi_period+1])
    
    for i in range(rsi_period+1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[rsi_period:] = 100 - (100 / (1 + rs[rsi_period:]))
    
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(bb_period, n):
        # Skip if data not ready
        if np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = df_1d['volume'].iloc[i]  # Current 1d volume
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_ma = vol_ma_1d_aligned[i]
        volume_confirm = volume > 1.5 * vol_ma
        
        # RSI filter: > 50 for long, < 50 for short
        rsi_val = rsi_aligned[i]
        rsi_long_filter = rsi_val > 50
        rsi_short_filter = rsi_val < 50
        
        if position == 0:
            if volume_confirm:
                # Long: price breaks above upper Bollinger Band and RSI > 50
                if price > upper_band_aligned[i] and rsi_long_filter:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below lower Bollinger Band and RSI < 50
                elif price < lower_band_aligned[i] and rsi_short_filter:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit when price crosses Bollinger middle band
            exit_signal = False
            
            if position == 1:  # long position
                if price < middle_band_aligned[i]:
                    exit_signal = True
            
            elif position == -1:  # short position
                if price > middle_band_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_BollingerBreakout_1wRSI_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0
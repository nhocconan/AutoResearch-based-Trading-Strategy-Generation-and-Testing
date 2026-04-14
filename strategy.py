#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day RSI with weekly Bollinger Bands for volatility regime.
# RSI(14) on daily timeframe identifies overbought/oversold conditions.
# Weekly Bollinger Bands (20,2) identify high/low volatility regimes.
# In low volatility (BB width < 50th percentile), mean revert at RSI extremes.
# In high volatility (BB width > 50th percentile), follow RSI momentum.
# Volume confirmation (>1.3x 20-period average) reduces false signals.
# Designed to work in both bull and bear markets by adapting to volatility regimes.
# Target: 20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Bollinger Bands (20,2)
    close_1w = df_1w['close'].values
    bb_length = 20
    bb_std = 2
    
    ma = pd.Series(close_1w).rolling(window=bb_length, min_periods=bb_length).mean().values
    std = pd.Series(close_1w).rolling(window=bb_length, min_periods=bb_length).std().values
    
    upper = ma + bb_std * std
    lower = ma - bb_std * std
    bb_width = upper - lower
    
    # Calculate 50th percentile of BB width for regime detection
    bb_width_50th = np.nanpercentile(bb_width, 50)
    
    # Align weekly Bollinger Bands to 4h timeframe
    ma_aligned = align_htf_to_ltf(prices, df_1w, ma)
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower)
    bb_width_aligned = align_htf_to_ltf(prices, df_1w, bb_width)
    bb_width_50th_aligned = align_htf_to_ltf(prices, df_1w, np.full_like(bb_width, bb_width_50th))
    
    # Load daily data ONCE for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate daily RSI(14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align daily RSI to 4h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume confirmation: 1.3x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 14, 20)  # Need BB, RSI, and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ma_aligned[i]) or 
            np.isnan(upper_aligned[i]) or
            np.isnan(lower_aligned[i]) or
            np.isnan(rsi_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # Regime detection: low volatility if BB width < 50th percentile
        low_volatility = bb_width_aligned[i] < bb_width_50th_aligned[i]
        high_volatility = bb_width_aligned[i] >= bb_width_50th_aligned[i]
        
        if position == 0:
            # Look for entries based on volatility regime
            if low_volatility:
                # Low volatility: mean revert at RSI extremes
                if (rsi_aligned[i] < 30 and 
                    volume_confirmed):
                    position = 1
                    signals[i] = position_size
                elif (rsi_aligned[i] > 70 and 
                      volume_confirmed):
                    position = -1
                    signals[i] = -position_size
            else:  # high_volatility
                # High volatility: follow RSI momentum
                if (rsi_aligned[i] > 50 and 
                    rsi_aligned[i] > rsi_aligned[i-1] and  # RSI rising
                    volume_confirmed):
                    position = 1
                    signals[i] = position_size
                elif (rsi_aligned[i] < 50 and 
                      rsi_aligned[i] < rsi_aligned[i-1] and  # RSI falling
                      volume_confirmed):
                    position = -1
                    signals[i] = -position_size
        elif position == 1:
            # Exit long: RSI returns to neutral (50) or opposite extreme
            if (rsi_aligned[i] >= 50 or 
                rsi_aligned[i] > 70):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI returns to neutral (50) or opposite extreme
            if (rsi_aligned[i] <= 50 or 
                rsi_aligned[i] < 30):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1wBB_1dRSI_VolatilityRegime_v1"
timeframe = "4h"
leverage = 1.0
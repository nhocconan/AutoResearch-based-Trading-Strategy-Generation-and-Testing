#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy combining 1d Donchian breakout with volume confirmation and 1w RSI trend filter.
# Uses Donchian(20) from 1d for support/resistance levels, providing robust breakout levels.
# Long when price breaks above 1d Donchian high with 1w RSI > 50 (uptrend) and volume confirmation.
# Short when price breaks below 1d Donchian low with 1w RSI < 50 (downtrend) and volume confirmation.
# Exit when price returns to 1d Donchian midpoint or RSI crosses 50 in opposite direction.
# Designed to work in both bull and bear markets by using Donchian channels for structure and RSI for trend confirmation.
# Target: 12-37 trades/year per symbol (50-150 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Donchian levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian(20) on 1d
    period = 20
    donchian_high = pd.Series(high_1d).rolling(window=period, min_periods=period).max().values
    donchian_low = pd.Series(low_1d).rolling(window=period, min_periods=period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Load 1w data ONCE for RSI trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate RSI(14) on 1w
    delta = np.diff(close_1w, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rs = np.where(avg_loss == 0, 100, rs)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Align indicators to lower timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Volume confirmation: 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20, 14)  # Need Donchian and RSI
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(donchian_mid_aligned[i]) or
            np.isnan(rsi_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: RSI > 50 for uptrend, < 50 for downtrend
        uptrend = rsi_1w_aligned[i] > 50
        downtrend = rsi_1w_aligned[i] < 50
        
        if position == 0:
            # Look for Donchian breakouts
            # Long: price breaks above Donchian high AND uptrend
            if (close[i] > donchian_high_aligned[i] and 
                uptrend and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low AND downtrend
            elif (close[i] < donchian_low_aligned[i] and 
                  downtrend and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian midpoint or RSI crosses below 50
            if (close[i] <= donchian_mid_aligned[i] or 
                rsi_1w_aligned[i] <= 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Donchian midpoint or RSI crosses above 50
            if (close[i] >= donchian_mid_aligned[i] or 
                rsi_1w_aligned[i] >= 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_DonchianBreakout_1wRSI_Volume_v1"
timeframe = "12h"
leverage = 1.0
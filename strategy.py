#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h RSI momentum and 1d Donchian breakout with volume confirmation.
# Long when price breaks above 1d Donchian high (20-period) AND 12h RSI > 60 (momentum) AND volume > 1.5x average.
# Short when price breaks below 1d Donchian low (20-period) AND 12h RSI < 40 (momentum) AND volume > 1.5x average.
# Exit when price returns to 1d midpoint or RSI crosses 50 in opposite direction.
# Designed to capture momentum bursts in both bull and bear markets with volume confirmation to reduce false breakouts.
# Target: 25-30 trades/year per symbol (100-120 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Load 12h data ONCE for RSI momentum
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate RSI(14) on 12h
    delta = np.diff(close_12h, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rs = np.where(avg_loss == 0, 100, rs)
    rsi_12h = 100 - (100 / (1 + rs))
    
    # Align indicators to lower timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1d, donch_mid)
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 14)  # Need Donchian and RSI
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or
            np.isnan(donch_mid_aligned[i]) or
            np.isnan(rsi_12h_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for Donchian breakouts with momentum confirmation
            # Long: price breaks above Donchian high AND bullish momentum
            if (close[i] > donch_high_aligned[i] and 
                rsi_12h_aligned[i] > 60 and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low AND bearish momentum
            elif (close[i] < donch_low_aligned[i] and 
                  rsi_12h_aligned[i] < 40 and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian midpoint or RSI drops below 50
            if (close[i] <= donch_mid_aligned[i] or 
                rsi_12h_aligned[i] < 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Donchian midpoint or RSI rises above 50
            if (close[i] >= donch_mid_aligned[i] or 
                rsi_12h_aligned[i] > 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1dDonchian_12hRSI_Momentum_v1"
timeframe = "4h"
leverage = 1.0
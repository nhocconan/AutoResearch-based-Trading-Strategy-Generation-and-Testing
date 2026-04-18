#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d RSI momentum filter and volume confirmation.
# Donchian breakouts capture momentum in trending markets, while RSI filter avoids overextended moves.
# Volume confirmation ensures breakouts have institutional participation.
# Designed for moderate trade frequency (20-40/year) to balance opportunity and fee drag.
# Works in bull markets (breakouts continue with trend) and bear markets (breakdowns continue with trend).
name = "4h_Donchian20_1dRSI_Volume"
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
    
    # Get daily data for RSI filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 20-period Donchian channels from previous periods (no look-ahead)
    # Upper = max(high over last 20 periods), Lower = min(low over last 20 periods)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 14-period RSI on daily close
    close_1d = pd.Series(df_1d['close'])
    delta = close_1d.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_values = rsi_14.values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper with RSI not overbought and volume
            if vol_confirm and close[i] > donchian_upper[i] and rsi_14_values[i // 16] < 70:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower with RSI not oversold and volume
            elif vol_confirm and close[i] < donchian_lower[i] and rsi_14_values[i // 16] > 30:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian lower (reversal signal)
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian upper (reversal signal)
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
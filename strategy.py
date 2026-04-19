#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h Donchian breakout + volume confirmation
# - 4h Donchian(20) defines trend direction and breakout levels
# - 1d volume > 1.3x 20-period average for conviction (reduced threshold for more signals)
# - 1h RSI(14) for entry timing: long when RSI < 35 in uptrend, short when RSI > 65 in downtrend
# - Exit on opposite RSI level (RSI > 65 for long, RSI < 35 for short) or Donchian reversal
# - Session filter: only trade 08:00-20:00 UTC to avoid low-volume periods
# - Position size: 0.20 (20%) to manage drawdown
# - Designed to work in both bull and bear markets by following higher timeframe trend
# - Target: 20-40 trades/year to avoid excessive fee drift

name = "1h_Donchian20_RSI_1dVolume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend and Donchian channels
    df_4h = get_htf_data(prices, '4h')
    
    # 4h Donchian(20) channels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 1h RSI(14) for entry timing
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], np.nan).fillna(0)  # Handle division by zero
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Pre-compute session filter (08:00-20:00 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(rsi_values[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 1d volume > 1.3x average
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.3 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Look for long entry: price breaks above 4h Donchian high + RSI not overbought + volume
            if close[i] > donchian_high_aligned[i] and rsi_values[i] < 65 and volume_filter:
                signals[i] = 0.20
                position = 1
            # Look for short entry: price breaks below 4h Donchian low + RSI not oversold + volume
            elif close[i] < donchian_low_aligned[i] and rsi_values[i] > 35 and volume_filter:
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long position: exit on RSI overbought or price breaks below Donchian low
            if rsi_values[i] > 65 or close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short position: exit on RSI oversold or price breaks above Donchian high
            if rsi_values[i] < 35 or close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
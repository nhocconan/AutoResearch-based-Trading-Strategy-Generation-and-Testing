#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d confluence with volume and session filter.
# Uses 4h trend direction via SMA50, 1d momentum via RSI(14), and 1h for entry timing.
# Long when: price > 4h SMA50, RSI(1d) > 50, and volume > 1.2x average (momentum long).
# Short when: price < 4h SMA50, RSI(1d) < 50, and volume > 1.2x average (momentum short).
# Entry only during active session (08-20 UTC) to avoid low-volume noise.
# Position size fixed at 0.20 to limit drawdown. Target: 15-30 trades/year.
# Works in bull/bear by following 4h trend and filtering with 1d momentum.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for trend (SMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Load 1d data for momentum (RSI)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 4h SMA50
    sma_50_4h = pd.Series(df_4h['close']).rolling(window=50, min_periods=50).mean().values
    
    # Calculate 1d RSI(14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = np.concatenate([[np.nan], rsi_1d])  # align with close_1d index
    
    # Align indicators to 1h timeframe
    sma_50_4h_aligned = align_htf_to_ltf(prices, df_4h, sma_50_4h)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume confirmation: 1.2x average volume (24-period on 1h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # Fixed 20% position
    
    # Start after enough data for calculations
    start = max(50, 24)  # Need SMA50 and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(sma_50_4h_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.2 * vol_ma[i]
        
        # Trend and momentum conditions
        price_above_sma = close[i] > sma_50_4h_aligned[i]
        price_below_sma = close[i] < sma_50_4h_aligned[i]
        rsi_bullish = rsi_1d_aligned[i] > 50
        rsi_bearish = rsi_1d_aligned[i] < 50
        
        if position == 0:
            # Look for entries
            if price_above_sma and rsi_bullish and volume_confirmed:
                position = 1
                signals[i] = position_size
            elif price_below_sma and rsi_bearish and volume_confirmed:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: trend/momentum reverses or volume drops
            if (price_below_sma or not rsi_bullish or not volume_confirmed):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: trend/momentum reverses or volume drops
            if (price_above_sma or not rsi_bearish or not volume_confirmed):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_1d_SMA_RSI_Volume_Session_v1"
timeframe = "1h"
leverage = 1.0
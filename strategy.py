#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1-day volume confirmation and RSI filter
# In bullish regime (price > 200-day EMA): long on upper band breakout + volume spike + RSI > 50
# In bearish regime (price < 200-day EMA): short on lower band breakout + volume spike + RSI < 50
# Uses 4h Donchian channels (20-period), 1-day volume for confirmation, 1-day RSI for regime filter
# Designed to work in both bull and bear markets by adapting to long-term trend
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for volume and RSI
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    close_1d = df_1d['close'].values
    
    # Calculate 200-day EMA for regime filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 14-period RSI on daily timeframe
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 20-period average volume on daily timeframe
    avg_vol_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    avg_vol_20_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_20)
    
    # Calculate 4h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    upper_channel = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if NaN in indicators
        if np.isnan(ema200_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(avg_vol_20_aligned[i]) or np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirm = volume > 1.5 * avg_vol_20_aligned[i]
        
        if position == 0:
            # Bullish regime: price above 200-day EMA
            if price > ema200_1d_aligned[i]:
                # Long on upper band breakout + volume confirmation + RSI > 50
                if price > upper_channel[i] and volume_confirm and rsi_1d_aligned[i] > 50:
                    signals[i] = 0.25
                    position = 1
            # Bearish regime: price below 200-day EMA
            elif price < ema200_1d_aligned[i]:
                # Short on lower band breakout + volume confirmation + RSI < 50
                if price < lower_channel[i] and volume_confirm and rsi_1d_aligned[i] < 50:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: price crosses below lower channel or RSI < 40
            if price < lower_channel[i] or rsi_1d_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above upper channel or RSI > 60
            if price > upper_channel[i] or rsi_1d_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeConfirm_RSIFilter"
timeframe = "4h"
leverage = 1.0
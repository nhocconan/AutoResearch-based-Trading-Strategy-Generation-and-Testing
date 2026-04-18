#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w RSI filter and volume confirmation.
# Donchian(20) breakout provides clear entry/exit signals.
# 1w RSI(14) > 60 for long and < 40 for short ensures we trade with momentum.
# Volume spike (>1.8x 20-period average) confirms conviction.
# Works in bull markets (breakouts up) and bear markets (breakouts down).
# Target: 7-25 trades/year (30-100 total over 4 years) to minimize fee drag.
name = "1d_Donchian20_1wRSI_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels on 1d data
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    
    donchian_high = high_1d.rolling(window=20, min_periods=20).max().values
    donchian_low = low_1d.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to lower timeframe (1d)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Get 1w data for RSI filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate RSI on 1w data
    close_1w = pd.Series(df_1w['close'].values)
    delta = close_1w.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    rs = avg_gain / avg_loss
    rsi_1w = (100 - (100 / (1 + rs))).values
    
    # Align RSI to lower timeframe (1d)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate volume spike: current volume > 1.8 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(rsi_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Price action
        close_val = close[i]
        
        # Donchian breakout conditions
        breakout_up = close_val > donchian_high_aligned[i]
        breakout_down = close_val < donchian_low_aligned[i]
        
        # RSI momentum filter
        rsi_val = rsi_1w_aligned[i]
        rsi_long_filter = rsi_val > 60
        rsi_short_filter = rsi_val < 40
        
        if position == 0:
            # Long: Breakout up AND RSI momentum (>60) AND volume spike
            if breakout_up and rsi_long_filter and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Breakout down AND RSI momentum (<40) AND volume spike
            elif breakout_down and rsi_short_filter and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Breakdown below lower Donchian band OR RSI weakens (<50)
            if close_val < donchian_low_aligned[i] or rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Breakout above upper Donchian band OR RSI weakens (>50)
            if close_val > donchian_high_aligned[i] or rsi_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
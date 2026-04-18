#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h RSI mean reversion with 1d volume filter and 1w ADX trend filter.
# RSI < 30 triggers long, RSI > 70 triggers short on 12h timeframe.
# Requires 1d volume > 1.5x 20-period average for confirmation.
# Requires 1w ADX > 20 to avoid choppy markets.
# Works in both bull and bear markets by fading extremes in the direction of higher timeframe trend.
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.
name = "12h_RSI_MeanReversion_1dVolume_1wADX"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for RSI calculation
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate RSI on 12h data
    close_12h = pd.Series(df_12h['close'].values)
    delta = close_12h.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_12h = 100 - (100 / (1 + rs))
    rsi_12h = rsi_12h.fillna(50).values  # Fill NaN with neutral 50
    
    # Align RSI to lower timeframe (12h)
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate volume filter on 1d data
    vol_1d = pd.Series(df_1d['volume'].values)
    vol_ma_20_1d = vol_1d.rolling(window=20, min_periods=20).mean().values
    vol_1d_array = vol_1d.values
    volume_filter_1d = vol_1d_array > (1.5 * vol_ma_20_1d)
    
    # Align volume filter to lower timeframe (12h)
    volume_filter_12h = align_htf_to_ltf(prices, df_1d, volume_filter_1d.astype(float))
    
    # Get 1w data for ADX filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate ADX on 1w data
    high_1w = pd.Series(df_1w['high'].values)
    low_1w = pd.Series(df_1w['low'].values)
    close_1w = pd.Series(df_1w['close'].values)
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = abs(high_1w - close_1w.shift(1))
    tr3 = abs(low_1w - close_1w.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = high_1w.diff()
    down_move = low_1w.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / atr_1w)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / atr_1w)
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1w = dx.ewm(alpha=1/14, adjust=False).mean().values
    
    # Align ADX to lower timeframe (12h)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_12h_aligned[i]) or 
            np.isnan(volume_filter_12h[i]) or
            np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        rsi_val = rsi_12h_aligned[i]
        vol_filter = volume_filter_12h[i] > 0.5  # Convert to boolean
        strong_trend = adx_1w_aligned[i] > 20
        
        if position == 0:
            # Long: RSI oversold AND volume confirmation AND trend filter
            if rsi_val < 30 and vol_filter and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought AND volume confirmation AND trend filter
            elif rsi_val > 70 and vol_filter and strong_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral OR volume dries up OR trend weakens
            if rsi_val > 50 or not vol_filter or adx_1w_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral OR volume dries up OR trend weakens
            if rsi_val < 50 or not vol_filter or adx_1w_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w ADX for trend strength and 1d RSI for mean reversion.
# 1w ADX > 20 filters for trending markets to avoid whipsaws in ranging conditions.
# 1d RSI < 30 or > 70 provides mean-reversion entries with price reversal expectation.
# Volume confirmation (>1.3x 20-period average) reduces false signals.
# Designed to work in both bull and bear markets by using 1w trend filter to avoid counter-trend trades.
# Target: 20-30 trades/year per symbol (80-120 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE for ADX calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate ADX on 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_period = 14
    atr = pd.Series(tr).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Calculate RSI on 1d data
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    rsi_period = 14
    avg_gain = pd.Series(gain).ewm(span=rsi_period, adjust=False, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, adjust=False, min_periods=rsi_period).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align indicators to daily timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume confirmation: 1.3x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 20)  # Need RSI and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # Trend filter: ADX > 20 indicates trending market
        trending = adx_aligned[i] > 20
        
        if position == 0:
            # Look for RSI mean reversion signals
            # Only trade in trending markets to avoid chop
            
            # Long: RSI oversold (<30) in uptrend
            if (rsi[i] < 30 and 
                trending and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: RSI overbought (>70) in downtrend
            elif (rsi[i] > 70 and 
                  trending and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral (50) or trend weakens
            if (rsi[i] >= 50 or 
                adx_aligned[i] < 15):  # Trend weakening
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI returns to neutral (50) or trend weakens
            if (rsi[i] <= 50 or 
                adx_aligned[i] < 15):  # Trend weakening
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1wADX_1dRSI_MeanReversion_v1"
timeframe = "1d"
leverage = 1.0
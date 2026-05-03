#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian upper channel in 1d uptrend with volume spike (>2.0x 20-period volume MA).
# Short when price breaks below Donchian lower channel in 1d downtrend with volume spike.
# Uses discrete position sizing (0.25) to minimize fee churn and ensure <200 total 4h trades over 4 years.
# Designed for 4h timeframe to achieve 75-200 total trades over 4 years (19-50/year) with Sharpe > 0 on BTC/ETH/SOL.

name = "4h_Donchian20_1dEMA50_VolumeSpike_ATR"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channel calculation
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels on 4h data
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Upper channel: 20-period high
    upper_channel = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Lower channel: 20-period low
    lower_channel = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to lower timeframe
    upper_channel_aligned = align_htf_to_ltf(prices, df_4h, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_4h, lower_channel)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detection (20-period volume MA on primary timeframe)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)  # Volume at least 2.0x average
    
    # ATR for stoploss (using 4h ATR)
    tr1 = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values - pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    tr2 = np.abs(pd.Series(high_4h).rolling(window=14, min_periods=14).max().values - pd.Series(close_4h).shift(1).rolling(window=14, min_periods=1).mean().values)
    tr3 = np.abs(pd.Series(low_4h).rolling(window=14, min_periods=14).min().values - pd.Series(close_4h).shift(1).rolling(window=14, min_periods=1).mean().values)
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # Track entry price for stoploss
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(upper_channel_aligned[i]) or np.isnan(lower_channel_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_4h_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper channel AND 1d uptrend AND volume spike
            if close_val > upper_channel_aligned[i] and close_val > ema_50_1d_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: price breaks below lower channel AND 1d downtrend AND volume spike
            elif close_val < lower_channel_aligned[i] and close_val < ema_50_1d_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            # Exit: price breaks below lower channel
            if close_val < lower_channel_aligned[i]:
                exit_signal = True
            # Exit: 1d trend changes to downtrend
            elif close_val < ema_50_1d_aligned[i]:
                exit_signal = True
            # Exit: ATR-based stoploss (2.0 * ATR below entry)
            elif close_val < entry_price - 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions for short
            exit_signal = False
            # Exit: price breaks above upper channel
            if close_val > upper_channel_aligned[i]:
                exit_signal = True
            # Exit: 1d trend changes to uptrend
            elif close_val > ema_50_1d_aligned[i]:
                exit_signal = True
            # Exit: ATR-based stoploss (2.0 * ATR above entry)
            elif close_val > entry_price + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
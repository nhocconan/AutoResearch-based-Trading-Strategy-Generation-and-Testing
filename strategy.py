#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian channel breakout with daily volatility filter and volume confirmation.
# Uses Donchian(20) breakouts on 12h timeframe, filtered by daily ATR volatility regime and volume spike.
# Designed for low trade frequency (target 15-30/year) to minimize fee drag in both bull and bear markets.
# Works in bull markets via breakout continuation and in bear markets via mean-reversion after volatility spikes.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR and Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on daily data with proper min_periods
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align daily ATR14 to 12h timeframe
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate Donchian(20) channels on daily data
    donch_high_20 = np.full(len(high_1d), np.nan)
    donch_low_20 = np.full(len(low_1d), np.nan)
    for i in range(20, len(high_1d)):
        donch_high_20[i] = np.max(high_1d[i-20:i])
        donch_low_20[i] = np.min(low_1d[i-20:i])
    
    # Align daily Donchian channels to 12h timeframe
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    
    # Calculate 12h ATR for stop loss and position sizing
    tr_12h_1 = high - low
    tr_12h_2 = np.abs(high - np.roll(close, 1))
    tr_12h_3 = np.abs(low - np.roll(close, 1))
    tr_12h_1[0] = high[0] - low[0]
    tr_12h_2[0] = np.abs(high[0] - close[0])
    tr_12h_3[0] = np.abs(low[0] - close[0])
    tr_12h = np.maximum(tr_12h_1, np.maximum(tr_12h_2, tr_12h_3))
    atr_12h = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # need daily Donchian20, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(donch_high_20_aligned[i]) or 
            np.isnan(donch_low_20_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when volatility is elevated (ATR > 1.5 * 20-period average)
        atr_ma_20 = np.full(len(atr_14_1d_aligned), np.nan)
        if i >= 20:
            atr_ma_20[i] = np.mean(atr_14_1d_aligned[i-20:i])
        vol_filter = (not np.isnan(atr_ma_20[i]) and 
                     atr_14_1d_aligned[i] > 1.5 * atr_ma_20[i])
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        vol_confirmed = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long entry: price breaks above daily Donchian high with volatility and volume confirmation
            if (close[i] > donch_high_20_aligned[i] and 
                vol_filter and 
                vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below daily Donchian low with volatility and volume confirmation
            elif (close[i] < donch_low_20_aligned[i] and 
                  vol_filter and 
                  vol_confirmed):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below daily Donchian low or ATR-based stop
            if (close[i] < donch_low_20_aligned[i] or 
                close[i] < open_price[i] - 2.0 * atr_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above daily Donchian high or ATR-based stop
            if (close[i] > donch_high_20_aligned[i] or 
                close[i] > open_price[i] + 2.0 * atr_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_DailyATR_VolumeFilter"
timeframe = "12h"
leverage = 1.0
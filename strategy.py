#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend (HTF)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly EMA(34) for trend
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get daily data for ATR and RSI (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR (14-period)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate daily RSI (14-period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    # Daily Donchian channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need weekly EMA, daily ATR/RSI, Donchian, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma20[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or 
            np.isnan(rsi_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # RSI filter: avoid extreme overbought/oversold
        rsi_filter = (rsi_14_aligned[i] > 30) and (rsi_14_aligned[i] < 70)
        
        # Trend filter: price above/below weekly EMA34
        trend_up = close[i] > ema_34_1w_aligned[i]
        trend_down = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume, RSI, and uptrend
            if (close[i] > highest_high[i] and volume_filter and rsi_filter and trend_up):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume, RSI, and downtrend
            elif (close[i] < lowest_low[i] and volume_filter and rsi_filter and trend_down):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to Donchian low or trend changes
            if close[i] < lowest_low[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to Donchian high or trend changes
            if close[i] > highest_high[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyEMA34_DailyATR_RSI_DonchianBreakout_Volume"
timeframe = "1d"
leverage = 1.0
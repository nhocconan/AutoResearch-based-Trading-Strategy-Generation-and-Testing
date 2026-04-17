#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 12-period EMA on daily closes for trend filter
    ema_12_1d = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_12_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_12_1d)
    
    # Calculate daily ATR for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 12-period EMA on current timeframe for entry signal
    ema_12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # Calculate 20-period volume average for confirmation
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # Need volume MA20 and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma20[i]) or 
            np.isnan(ema_12[i]) or 
            np.isnan(ema_12_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3 * 20-period average
        volume_filter = volume[i] > (1.3 * volume_ma20[i])
        
        # Trend filter: price above/below daily EMA12
        above_trend = close[i] > ema_12_1d_aligned[i]
        below_trend = close[i] < ema_12_1d_aligned[i]
        
        # Momentum filter: price above/below 12-period EMA on current timeframe
        above_ema = close[i] > ema_12[i]
        below_ema = close[i] < ema_12[i]
        
        if position == 0:
            # Long: price above daily EMA12, above EMA12, with volume
            if above_trend and above_ema and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: price below daily EMA12, below EMA12, with volume
            elif below_trend and below_ema and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below daily EMA12 or EMA12
            if not above_trend or not above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above daily EMA12 or EMA12
            if not below_trend or not below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_DailyEMA_Momentum_Volume"
timeframe = "12h"
leverage = 1.0
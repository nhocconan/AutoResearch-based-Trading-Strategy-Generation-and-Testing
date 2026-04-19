#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_Trend_With_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend analysis
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA (34 period) for trend direction
    weekly_close = df_weekly['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate weekly ATR for volatility filtering
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close_arr = df_weekly['close'].values
    
    tr1 = weekly_high - weekly_low
    tr2 = np.abs(weekly_high - np.roll(weekly_close_arr, 1))
    tr3 = np.abs(weekly_low - np.roll(weekly_close_arr, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    weekly_atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate daily volume average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    # Align weekly indicators to daily timeframe
    weekly_ema_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema)
    weekly_atr_aligned = align_htf_to_ltf(prices, df_weekly, weekly_atr)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(weekly_ema_aligned[i]) or np.isnan(weekly_atr_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Trend condition: price relative to weekly EMA
        price_above_ema = close[i] > weekly_ema_aligned[i]
        price_below_ema = close[i] < weekly_ema_aligned[i]
        
        # Volatility filter: avoid extremely volatile conditions
        volatility_filter = weekly_atr_aligned[i] > 0  # Always true if ATR calculated
        
        if position == 0:
            # Long when price above weekly EMA + volume spike
            if price_above_ema and volume_spike[i] and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short when price below weekly EMA + volume spike
            elif price_below_ema and volume_spike[i] and volatility_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price crosses below weekly EMA
            if price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price crosses above weekly EMA
            if price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
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
    
    # Get daily data for ATR and price levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate daily ATR (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily moving averages for trend
    sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma200_1d = pd.Series(close_1d).rolling(window=200, min_periods=200).mean().values
    
    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily indicators to 1d timeframe (no shift needed as we're already on 1d)
    atr_aligned = atr_14
    sma50_aligned = sma50_1d
    sma200_aligned = sma200_1d
    
    # Align weekly EMA to 1d timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: above average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Hour filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for sufficient warmup (for SMA200)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_aligned[i]) or np.isnan(sma50_aligned[i]) or 
            np.isnan(sma200_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: above average volume
        vol_filter = volume[i] > vol_ma[i]
        
        # Trend filter: price above both SMAs and weekly EMA
        bullish_alignment = (close[i] > sma50_aligned[i]) and (sma50_aligned[i] > sma200_aligned[i]) and (close[i] > ema50_1w_aligned[i])
        bearish_alignment = (close[i] < sma50_aligned[i]) and (sma50_aligned[i] < sma200_aligned[i]) and (close[i] < ema50_1w_aligned[i])
        
        # Volatility filter: avoid extremely low volatility days
        vol_filter_atr = atr_aligned[i] > 0.01 * close[i]  # ATR > 1% of price
        
        # Entry conditions: 
        # Long: price above rising SMAs with volume and bullish alignment
        # Short: price below falling SMAs with volume and bearish alignment
        long_entry = bullish_alignment and vol_filter and vol_filter_atr
        short_entry = bearish_alignment and vol_filter and vol_filter_atr
        
        # Exit conditions: trend reversal or volatility spike
        long_exit = not bullish_alignment
        short_exit = not bearish_alignment
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_SMA_Alignment_Trend_Filter_Volume_Session"
timeframe = "1d"
leverage = 1.0
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
    
    # Get daily data for pivot calculation and volume analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate daily ATR for volatility filtering
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range components for daily ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily moving averages for trend context
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    
    # Align daily indicators to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_20_1d)
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Calculate volume ratio (current volume vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Time filters
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(sma_20_1d_aligned[i]) or 
            np.isnan(sma_50_1d_aligned[i]) or np.isnan(ema_20_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC (most active hours)
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
        
        # Volatility filter: avoid extremely low volatility periods
        vol_filter = atr_1d_aligned[i] > 0
        
        # Volume filter: above average volume
        vol_filter = vol_filter and (volume[i] > vol_ma_20[i])
        
        # Trend filter: price above/below 12h EMA20
        trend_up = close[i] > ema_20_12h_aligned[i]
        trend_down = close[i] < ema_20_12h_aligned[i]
        
        # Market regime filter: avoid ranging markets
        # Use price position relative to SMAs to detect trending conditions
        price_above_sma20 = close[i] > sma_20_1d_aligned[i]
        price_below_sma50 = close[i] < sma_50_1d_aligned[i]
        
        # Strong uptrend: price above both SMAs
        strong_uptrend = price_above_sma20 and (close[i] > sma_50_1d_aligned[i])
        # Strong downtrend: price below both SMAs
        strong_downtrend = price_below_sma50 and (close[i] < sma_20_1d_aligned[i])
        
        # Entry conditions:
        # Long: strong uptrend + volume + session
        # Short: strong downtrend + volume + session
        long_entry = strong_uptrend and vol_filter and trend_up
        short_entry = strong_downtrend and vol_filter and trend_down
        
        # Exit conditions: trend reversal or volatility expansion
        long_exit = not strong_uptrend or (atr_1d_aligned[i] < 0.5 * atr_1d_aligned[max(0, i-1)])
        short_exit = not strong_downtrend or (atr_1d_aligned[i] < 0.5 * atr_1d_aligned[max(0, i-1)])
        
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

name = "4h_StrongTrend_EMA20_Volume_Session_Filter"
timeframe = "4h"
leverage = 1.0
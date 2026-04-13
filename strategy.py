#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and 1w trend filter
    # Long: price breaks above H3 resistance + volume > 2x 20-period average + 1w close > 1w EMA20
    # Short: price breaks below L3 support + volume > 2x 20-period average + 1w close < 1w EMA20
    # Uses discrete sizing (0.25) to minimize fee drag and ATR-based stoploss
    # Target: 12-37 trades/year to stay within 12h optimal range
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day)
    # H3 = Close + 1.1 * (High - Low) / 2
    # L3 = Close - 1.1 * (High - Low) / 2
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Calculate 1d volume average (20-period) for confirmation
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA20 for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    atr_1d = np.zeros(n)  # ATR using daily true range
    
    # Calculate ATR (daily true range)
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = close_1d[0]
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - prev_close_1d)
    tr3 = np.abs(low_1d - prev_close_1d)
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d_raw = pd.Series(true_range).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    for i in range(n):
        idx_1d = i // 2  # 12h bars in 1d timeframe (2 bars per day)
        if idx_1d < len(atr_1d_raw):
            atr_1d[i] = atr_1d_raw[idx_1d]
        else:
            atr_1d[i] = 0
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        idx_1d = i // 2  # 12h bars in 1d timeframe (2 bars per day)
        if idx_1d >= len(volume_1d):
            signals[i] = 0.0
            continue
            
        # Volume confirmation: current 1d volume > 2x 20-period average
        volume_confirmed = volume_1d[idx_1d] > 2.0 * vol_avg_20_1d_aligned[i]
        
        # Trend filter: 1w close above/below EMA20
        if idx_1d < len(close_1d):
            uptrend = close_1d[idx_1d] > ema_20_1w_aligned[i]
            downtrend = close_1d[idx_1d] < ema_20_1w_aligned[i]
        else:
            uptrend = False
            downtrend = False
        
        # Breakout conditions: price breaks Camarilla levels with volume and trend
        breakout_long = (close[i] > camarilla_h3_aligned[i]) and volume_confirmed and uptrend
        breakout_short = (close[i] < camarilla_l3_aligned[i]) and volume_confirmed and downtrend
        
        # Stoploss: 2.0x ATR below/above entry
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - 2.0 * atr_1d[i]
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + 2.0 * atr_1d[i]
        
        # Execute signals
        if breakout_long and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            elif position == -1:
                signals[i] = -position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            else:
                signals[i] = 0.0
                entry_price[i] = np.nan
    
    return signals

name = "12h_1d_1w_camarilla_volume_trend_v1"
timeframe = "12h"
leverage = 1.0
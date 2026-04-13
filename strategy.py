#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and 1d trend filter
    # Long: price breaks above Camarilla H3 level + volume > 1.3x 20-period average + 1d close > 1d EMA20
    # Short: price breaks below Camarilla L3 level + volume > 1.3x 20-period average + 1d close < 1d EMA20
    # Uses discrete sizing (0.25) to minimize fee drag and ATR-based stoploss
    # Target: 20-40 trades/year to stay within 4h optimal range
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Get 1d data for Camarilla calculation, volume confirmation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    camarilla_h3 = np.zeros(len(high_1d))
    camarilla_l3 = np.zeros(len(low_1d))
    camarilla_h4 = np.zeros(len(high_1d))
    camarilla_l4 = np.zeros(len(low_1d))
    
    for i in range(1, len(high_1d)):
        # Calculate pivot point and range from previous day
        pp = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3.0
        rng = high_1d[i-1] - low_1d[i-1]
        
        # Camarilla levels
        camarilla_h3[i] = pp + rng * 1.1 / 4
        camarilla_l3[i] = pp - rng * 1.1 / 4
        camarilla_h4[i] = pp + rng * 1.1 / 2
        camarilla_l4[i] = pp - rng * 1.1 / 2
    
    # Calculate 1d volume average (20-period) for confirmation
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA20 for trend filter
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    atr_1d = np.zeros(n)  # ATR using daily true range
    
    # Calculate ATR (true range) for stoploss
    for i in range(n):
        idx_1d = i // 6  # 4h bars in 1d timeframe (6 bars per day)
        if idx_1d < len(high_1d) and idx_1d < len(low_1d) and idx_1d < len(close_1d):
            if idx_1d == 0:
                tr = high_1d[idx_1d] - low_1d[idx_1d]
            else:
                tr1 = high_1d[idx_1d] - low_1d[idx_1d]
                tr2 = abs(high_1d[idx_1d] - close_1d[idx_1d-1])
                tr3 = abs(low_1d[idx_1d] - close_1d[idx_1d-1])
                tr = max(tr1, tr2, tr3)
            atr_1d[i] = tr
        else:
            atr_1d[i] = 0
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(ema_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period average
        idx_1d = i // 6  # 4h bars in 1d timeframe (6 bars per day)
        if idx_1d >= len(volume_1d):
            signals[i] = 0.0
            continue
        volume_confirmed = volume_1d[idx_1d] > 1.3 * vol_avg_20_1d_aligned[i]
        
        # Trend filter: 1d close above/below EMA20
        uptrend = close_1d[idx_1d] > ema_20_1d_aligned[i] if idx_1d < len(close_1d) else False
        downtrend = close_1d[idx_1d] < ema_20_1d_aligned[i] if idx_1d < len(close_1d) else False
        
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

name = "4h_1d_camarilla_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0
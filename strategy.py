#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Camarilla pivot breakout with 1w volume confirmation and trend filter
    # Long: price breaks above Camarilla H3 + volume > 1.5x 20-period average + 1w close > 1w EMA20
    # Short: price breaks below Camarilla L3 + volume > 1.5x 20-period average + 1w close < 1w EMA20
    # Uses discrete sizing (0.25) to minimize fee drag and ATR-based stoploss
    # Target: 10-25 trades/year to stay within 1d optimal range
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get 1w data for trend filter and volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1d Camarilla levels (based on previous day)
    # H3 = close + 1.1*(high - low)/2
    # L3 = close - 1.1*(high - low)/2
    # Using previous day's values to avoid look-ahead
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_h3 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 2
    camarilla_l3 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 2
    
    # Calculate 1d volume average (20-period) for confirmation
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA20 for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1w volume average (20-period) for confirmation
    vol_avg_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 1d timeframe (prices is already 1d)
    camarilla_h3_aligned = camarilla_h3  # Already aligned to 1d
    camarilla_l3_aligned = camarilla_l3  # Already aligned to 1d
    vol_avg_20_1d_aligned = vol_avg_20_1d  # Already aligned to 1d
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    vol_avg_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    atr_1d = np.zeros(n)  # ATR using daily range
    
    # Calculate ATR (daily range) for stoploss
    for i in range(n):
        if i < len(high_1d) and i < len(low_1d):
            daily_range = high_1d[i] - low_1d[i]
            atr_1d[i] = daily_range * 0.5  # Approximate ATR as 50% of daily range
        else:
            atr_1d[i] = 0
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(vol_avg_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average OR 1w volume > 1.5x 20-period average
        volume_confirmed_1d = volume_1d[i] > 1.5 * vol_avg_20_1d_aligned[i]
        volume_confirmed_1w = volume_1w[i // 7] > 1.5 * vol_avg_20_1w_aligned[i] if i // 7 < len(volume_1w) else False
        volume_confirmed = volume_confirmed_1d or volume_confirmed_1w
        
        # Trend filter: 1w close above/below EMA20
        uptrend = close_1d[i] > ema_20_1w_aligned[i]
        downtrend = close_1d[i] < ema_20_1w_aligned[i]
        
        # Breakout conditions: price breaks Camarilla levels with volume and trend
        breakout_long = (close[i] > camarilla_h3_aligned[i]) and volume_confirmed and uptrend
        breakout_short = (close[i] < camarilla_l3_aligned[i]) and volume_confirmed and downtrend
        
        # Stoploss: 1.5x ATR below/above entry
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - 1.5 * atr_1d[i]
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + 1.5 * atr_1d[i]
        
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

name = "1d_1w_camarilla_breakout_volume_trend_v1"
timeframe = "1d"
leverage = 1.0
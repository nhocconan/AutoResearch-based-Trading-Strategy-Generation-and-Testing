#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Hypothesis: 1d Camarilla pivot breakout with 1w volume confirmation and 1w trend filter
    # Long: price breaks above H3 level + volume > 1.5x 20-period 1w average + 1w close > 1w EMA50
    # Short: price breaks below L3 level + volume > 1.5x 20-period 1w average + 1w close < 1w EMA50
    # Uses discrete sizing (0.25) to minimize fee drag and ATR-based stoploss
    # Target: 7-25 trades/year to stay within 1d optimal range
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Camarilla pivots, volume confirmation, and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w Camarilla levels (based on previous week)
    # Pivot = (H+L+C)/3
    # H3 = Pivot + 1.1*(H-L)
    # L3 = Pivot - 1.1*(H-L)
    pivot = (high_1w + low_1w + close_1w) / 3.0
    hl_range = high_1w - low_1w
    h3 = pivot + 1.1 * hl_range
    l3 = pivot - 1.1 * hl_range
    
    # Calculate 1w volume average (20-period) for confirmation
    vol_avg_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 1d timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1w, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1w, l3)
    vol_avg_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_20_1w)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    # Calculate ATR using true range approximation for 1d timeframe
    atr_1d = np.zeros(n)
    for i in range(1, n):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        if i < 14:
            atr_1d[i] = tr  # Simple average for warmup
        else:
            atr_1d[i] = 0.93 * atr_1d[i-1] + 0.07 * tr  # Wilder's smoothing
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or
            np.isnan(vol_avg_20_1w_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period 1w average
        vol_avg_20_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        if np.isnan(vol_avg_20_1d[i]):
            signals[i] = 0.0
            continue
        volume_confirmed = volume[i] > 1.5 * vol_avg_20_1w_aligned[i]
        
        # Trend filter: 1d close above/below 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Breakout conditions: price breaks Camarilla levels with volume and trend
        breakout_long = (close[i] > h3_aligned[i]) and volume_confirmed and uptrend
        breakout_short = (close[i] < l3_aligned[i]) and volume_confirmed and downtrend
        
        # Stoploss: 2x ATR below/above entry
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

name = "1d_1w_camarilla_volume_trend_v2"
timeframe = "1d"
leverage = 1.0
# 1d_TripleScreen_ElderForce_TrendFollow
# Triple Screen system: 1w trend filter (Elder Force Index), 1d entry trigger (Force Index divergence), volume confirmation
# Designed for trend following in both bull and bear markets with controlled trade frequency
# Target: 15-25 trades/year per symbol

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for primary calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Elder Force Index (13-period) on 1d
    price_change = np.diff(close_1d, prepend=close_1d[0])
    force_raw = price_change * volume_1d
    force_index = pd.Series(force_raw).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d EMA(13) for trend context
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Load 1w data for trend filter (Elder Force Index)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly Elder Force Index (13-period)
    price_change_w = np.diff(close_1w, prepend=close_1w[0])
    force_raw_w = price_change_w * volume_1w
    force_index_w = pd.Series(force_raw_w).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Weekly EMA(13) for additional trend confirmation
    ema_13_w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Align weekly indicators to daily timeframe
    force_index_w_aligned = align_htf_to_ltf(prices, df_1w, force_index_w)
    ema_13_w_aligned = align_htf_to_ltf(prices, df_1w, ema_13_w)
    
    # Daily price and volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.3x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20) > 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if NaN in critical values
        if (np.isnan(force_index[i]) or np.isnan(ema_13[i]) or
            np.isnan(force_index_w_aligned[i]) or np.isnan(ema_13_w_aligned[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        force_today = force_index[i]
        ema_today = ema_13[i]
        force_weekly = force_index_w_aligned[i]
        ema_weekly = ema_13_w_aligned[i]
        vol_ok = vol_filter[i]
        
        # Trend conditions: weekly force and EMA alignment
        bullish_trend = force_weekly > 0 and price > ema_weekly
        bearish_trend = force_weekly < 0 and price < ema_weekly
        
        # Entry signals: daily force crossing zero with trend alignment
        bullish_entry = force_today > 0 and force_index[i-1] <= 0 and bullish_trend and vol_ok
        bearish_entry = force_today < 0 and force_index[i-1] >= 0 and bearish_trend and vol_ok
        
        # Exit signals: force reverses or trend breaks
        bullish_exit = force_today < 0 or (price < ema_today and force_weekly < 0)
        bearish_exit = force_today > 0 or (price > ema_today and force_weekly > 0)
        
        if position == 0:
            if bullish_entry:
                signals[i] = 0.25
                position = 1
            elif bearish_entry:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            if bullish_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if bearish_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_TripleScreen_ElderForce_TrendFollow"
timeframe = "1d"
leverage = 1.0
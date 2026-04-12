#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Camarilla pivot breakout with 1w trend filter and volume confirmation
    # Long when price breaks above H3 with volume > 1.5x 20-day average and 1w uptrend
    # Short when price breaks below L3 with volume > 1.5x 20-day average and 1w downtrend
    # Exit when price returns to pivot point (PP) or opposite Camarilla level
    # Discrete position sizing (0.25) to minimize fee churn. Target: 20-50 trades/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(34) for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate daily Camarilla levels using previous day's OHLC
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # PP = (H + L + C) / 3
    # H3 = PP + (H - L) * 1.1 / 4
    # L3 = PP - (H - L) * 1.1 / 4
    
    # Shift to get previous day's OHLC
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # First bar has no previous day
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan
    
    pivot_point = (prev_high + prev_low + prev_close) / 3.0
    high_low_range = prev_high - prev_low
    
    camarilla_h3 = pivot_point + (high_low_range * 1.1 / 4.0)
    camarilla_l3 = pivot_point - (high_low_range * 1.1 / 4.0)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    vol_ratio = np.full(n, np.nan)
    for i in range(20, n):
        if vol_ma_20[i] > 0:
            vol_ratio[i] = volume[i] / vol_ma_20[i]
        else:
            vol_ratio[i] = 1.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(pivot_point[i]) or np.isnan(camarilla_h3[i]) or 
            np.isnan(camarilla_l3[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from 1w EMA(34)
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Breakout conditions
        breakout_above_h3 = close[i] > camarilla_h3[i]
        breakout_below_l3 = close[i] < camarilla_l3[i]
        
        # Entry conditions with volume confirmation
        long_entry = breakout_above_h3 and (vol_ratio[i] > 1.5) and uptrend
        short_entry = breakout_below_l3 and (vol_ratio[i] > 1.5) and downtrend
        
        # Exit conditions: price returns to pivot point
        long_exit = close[i] < pivot_point[i]
        short_exit = close[i] > pivot_point[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_camarilla_breakout_vol_trend_v1"
timeframe = "1d"
leverage = 1.0
# 4h_1d_camarilla_breakout_volume_filter
# Hypothesis: Camarilla pivot levels on daily chart act as strong support/resistance zones.
# A breakout above/below these levels with volume confirmation signals strong momentum.
# Uses 12h trend filter to avoid counter-trend trades. Works in both bull (breakouts continue) 
# and bear (breakdowns continue) markets by trading with the higher timeframe trend.
# Target: 20-40 trades/year to minimize fee drag.

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
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    # H4 = Close + 1.1/2 * (High - Low) = Close + 0.55 * (High - Low)
    # L4 = Close - 1.1/2 * (High - Low) = Close - 0.55 * (High - Low)
    # H3 = Close + 1.1/4 * (High - Low) = Close + 0.275 * (High - Low)
    # L3 = Close - 1.1/4 * (High - Low) = Close - 0.275 * (High - Low)
    # H2 = Close + 1.1/6 * (High - Low) = Close + 0.1833 * (High - Low)
    # L2 = Close - 1.1/6 * (High - Low) = Close - 0.1833 * (High - Low)
    # H1 = Close + 1.1/12 * (High - Low) = Close + 0.0916 * (High - Low)
    # L1 = Close - 1.1/12 * (High - Low) = Close - 0.0916 * (High - Low)
    
    # Calculate for previous day (shifted by 1)
    high_prev = np.roll(high_1d, 1)
    low_prev = np.roll(low_1d, 1)
    close_prev = np.roll(close_1d, 1)
    high_prev[0] = high_prev[1] if len(high_prev) > 1 else high_prev[0]
    low_prev[0] = low_prev[1] if len(low_prev) > 1 else low_prev[0]
    close_prev[0] = close_prev[1] if len(close_prev) > 1 else close_prev[0]
    
    rangeprev = high_prev - low_prev
    H4 = close_prev + 0.55 * rangeprev
    L4 = close_prev - 0.55 * rangeprev
    H3 = close_prev + 0.275 * rangeprev
    L3 = close_prev - 0.275 * rangeprev
    H2 = close_prev + 0.1833 * rangeprev
    L2 = close_prev - 0.1833 * rangeprev
    H1 = close_prev + 0.0916 * rangeprev
    L1 = close_prev - 0.0916 * rangeprev
    
    # Get 12h trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # EMA 20 on 12h for trend
    close_12h_series = pd.Series(close_12h)
    ema_20_12h = close_12h_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all to 4h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Volume confirmation: volume > 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or 
            np.isnan(ema_20_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter
        uptrend = close[i] > ema_20_12h_aligned[i]
        downtrend = close[i] < ema_20_12h_aligned[i]
        
        # Breakout conditions with volume confirmation
        long_breakout = (close[i] > H4_aligned[i]) and vol_confirm[i]
        short_breakout = (close[i] < L4_aligned[i]) and vol_confirm[i]
        
        # Exit when price returns to opposite level or trend changes
        exit_long = position == 1 and (close[i] < L3_aligned[i] or not uptrend)
        exit_short = position == -1 and (close[i] > H3_aligned[i] or not downtrend)
        
        # Execute signals
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_breakout_volume_filter"
timeframe = "4h"
leverage = 1.0
# 1d_1W_Camarilla_Pivot_Breakout_v1
# Hypothesis: Camarilla pivot levels on 1-day chart provide strong support/resistance levels.
# Breakouts from these levels with volume confirmation and weekly trend filter capture
# momentum in both bull and bear markets. Weekly trend filter ensures we only trade
# in the direction of the higher timeframe trend, reducing false signals.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels on daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    sma_1w = np.zeros(len(close_1w))
    for i in range(20, len(close_1w)):
        sma_1w[i] = np.mean(close_1w[i-20:i])
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(1, n):
        # Need previous day's data for Camarilla calculation
        if i == 0:
            signals[i] = 0.0
            continue
            
        # Get previous day's OHLC (using daily data)
        # Find the index of the previous day in 1d data
        prev_day_idx = len(df_1d) - 1
        # Simple approach: use last available daily data
        if len(df_1d) >= 2:
            prev_high = df_1d['high'].iloc[-2]
            prev_low = df_1d['low'].iloc[-2]
            prev_close = df_1d['close'].iloc[-2]
        else:
            signals[i] = 0.0
            continue
            
        # Calculate Camarilla levels
        range_val = prev_high - prev_low
        if range_val <= 0:
            signals[i] = 0.0
            continue
            
        # Camarilla levels
        L4 = prev_close - (range_val * 1.1 / 2)
        L3 = prev_close - (range_val * 1.1 / 4)
        L2 = prev_close - (range_val * 1.1 / 6)
        L1 = prev_close - (range_val * 1.1 / 12)
        H1 = prev_close + (range_val * 1.1 / 12)
        H2 = prev_close + (range_val * 1.1 / 6)
        H3 = prev_close + (range_val * 1.1 / 4)
        H4 = prev_close + (range_val * 1.1 / 2)
        
        price = close[i]
        vol = volume[i]
        
        # Volume confirmation: current volume > 1.5x average volume (20-period)
        if i >= 20:
            avg_volume = np.mean(volume[i-20:i])
            volume_confirm = vol > 1.5 * avg_volume
        else:
            volume_confirm = False
            
        # Weekly trend filter
        weekly_trend_up = sma_1w_aligned[i] > 0 if not np.isnan(sma_1w_aligned[i]) else False
        weekly_trend_down = sma_1w_aligned[i] < 0 if not np.isnan(sma_1w_aligned[i]) else False
        
        if position == 0:
            # Long: price breaks above H3 or H4 with volume and weekly uptrend
            if (price > H3 and volume_confirm and weekly_trend_up):
                position = 1
                signals[i] = position_size
            # Short: price breaks below L3 or L4 with volume and weekly downtrend
            elif (price < L3 and volume_confirm and weekly_trend_down):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below L3 or weekly trend turns down
            if (price < L3 or (not weekly_trend_up and weekly_trend_down)):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above H3 or weekly trend turns up
            if (price > H3 or (not weekly_trend_down and weekly_trend_up)):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1W_Camarilla_Pivot_Breakout_v1"
timeframe = "1d"
leverage = 1.0
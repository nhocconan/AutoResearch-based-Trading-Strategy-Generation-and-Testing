#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout with 1d trend filter and volume confirmation
    # Long when: price breaks above Camarilla H3 AND price > 1d EMA(50) (uptrend) AND volume > 1.3x 20-bar avg volume
    # Short when: price breaks below Camarilla L3 AND price < 1d EMA(50) (downtrend) AND volume > 1.3x 20-bar avg volume
    # Exit when: price crosses Camarilla pivot point (PP) OR adverse 1d EMA(50) crossover
    # Uses discrete sizing (0.25) targeting 12-37 trades/year on 12h timeframe.
    # Works in bull/bear via 1d EMA(50) trend filter preventing counter-trend trades.
    # Volume confirmation reduces false breakouts in choppy markets.
    # Camarilla pivots provide strong intraday support/resistance levels.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 1d EMA(50)
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Camarilla pivot levels from previous 12h bar
    # Camarilla formulas: PP = (H+L+C)/3, Range = H-L
    # H4 = PP + Range * 1.1/2, L4 = PP - Range * 1.1/2
    # H3 = PP + Range * 1.1/4, L3 = PP - Range * 1.1/4
    # H2 = PP + Range * 1.1/6, L2 = PP - Range * 1.1/6
    # H1 = PP + Range * 1.1/12, L1 = PP - Range * 1.1/12
    
    # Shift high/low/close by 1 to use previous bar for pivot calculation
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot_point = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    camarilla_h3 = pivot_point + range_val * 1.1 / 4
    camarilla_l3 = pivot_point - range_val * 1.1 / 4
    camarilla_pp = pivot_point  # Exit level
    
    # Calculate volume confirmation: volume > 1.3x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.3 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(pivot_point[i]) or np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions
        breakout_up = close[i] > camarilla_h3[i-1]  # Break above previous H3
        breakout_down = close[i] < camarilla_l3[i-1]  # Break below previous L3
        
        # 1d EMA(50) trend filter
        uptrend = close[i] > ema_1d_aligned[i]
        downtrend = close[i] < ema_1d_aligned[i]
        
        # Entry conditions with volume confirmation
        long_entry = breakout_up and uptrend and volume_confirmed[i] and position != 1
        short_entry = breakout_down and downtrend and volume_confirmed[i] and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and (close[i] < camarilla_pp[i] or not uptrend))
        exit_short = (position == -1 and (close[i] > camarilla_pp[i] or not downtrend))
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
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

name = "12h_1d_camarilla_ema_volume_v1"
timeframe = "12h"
leverage = 1.0
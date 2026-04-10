#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h/1d trend filters and session filter
# - Long when price breaks above Camarilla H3 level with 4h close > 4h EMA50 AND 1d close > 1d EMA50
# - Short when price breaks below Camarilla L3 level with 4h close < 4h EMA50 AND 1d close < 1d EMA50
# - Exit when price retreats to Camarilla H4/L4 levels
# - Session filter: only trade 08:00-20:00 UTC to avoid low-volume periods
# - Position size: 0.20 (20% of capital) to limit drawdown
# - Target: 15-37 trades/year (60-150 total over 4 years) by using tight entry conditions
# - Uses multi-timeframe alignment: 4h/1d for direction, 1h for timing precision
# - Designed to work in both bull (trend following) and bear (counter-trend bounces) markets

name = "1h_4h_1d_camarilla_breakout_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute session filter (08:00-20:00 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Pre-compute 1h data
    h_1h = prices['high'].values
    l_1h = prices['low'].values
    c_1h = prices['close'].values
    
    # Pre-compute aligned 4h data properly
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    c_4h = df_4h['close'].values
    
    h_4h_aligned = align_htf_to_ltf(prices, df_4h, h_4h)
    l_4h_aligned = align_htf_to_ltf(prices, df_4h, l_4h)
    c_4h_aligned = align_htf_to_ltf(prices, df_4h, c_4h)
    
    # Pre-compute aligned 1d data properly
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    h_1d_aligned = align_htf_to_ltf(prices, df_1d, h_1d)
    l_1d_aligned = align_htf_to_ltf(prices, df_1d, l_1d)
    c_1d_aligned = align_htf_to_ltf(prices, df_1d, c_1d)
    
    # Pre-compute 4h EMA(50) for trend filter
    ema50_4h = pd.Series(c_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Pre-compute 1d EMA(50) for trend filter
    ema50_1d = pd.Series(c_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(h_4h_aligned[i]) or np.isnan(l_4h_aligned[i]) or np.isnan(c_4h_aligned[i]) or
            np.isnan(h_1d_aligned[i]) or np.isnan(l_1d_aligned[i]) or np.isnan(c_1d_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Skip if outside trading session
        if not in_session.iloc[i]:
            # Hold current position or flat outside session
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Get previous completed 1h bar values (need to shift by 1 to avoid look-ahead)
        if i >= 1:
            ph = h_1h[i-1]  # Previous 1h bar's high
            pl = l_1h[i-1]  # Previous 1h bar's low
            pc = c_1h[i-1]  # Previous 1h bar's close
            
            # Calculate Camarilla levels
            range_val = ph - pl
            if range_val > 0:
                camarilla_h3 = pc + (range_val * 1.1 / 4)
                camarilla_l3 = pc - (range_val * 1.1 / 4)
                camarilla_h4 = pc + (range_val * 1.1 / 2)
                camarilla_l4 = pc - (range_val * 1.1 / 2)
                
                if position == 0:  # Flat - look for new breakout entries
                    # Long breakout: price > Camarilla H3 with 4h uptrend AND 1d uptrend
                    if (c_1h[i] > camarilla_h3 and 
                        c_4h_aligned[i] > ema50_4h_aligned[i] and 
                        c_1d_aligned[i] > ema50_1d_aligned[i]):
                        position = 1
                        signals[i] = 0.20
                    # Short breakdown: price < Camarilla L3 with 4h downtrend AND 1d downtrend
                    elif (c_1h[i] < camarilla_l3 and 
                          c_4h_aligned[i] < ema50_4h_aligned[i] and 
                          c_1d_aligned[i] < ema50_1d_aligned[i]):
                        position = -1
                        signals[i] = -0.20
                else:  # Have position - look for exit
                    # Exit when price retreats to Camarilla H4/L4 levels
                    if position == 1:  # Long position
                        if c_1h[i] < camarilla_h4:
                            position = 0
                            signals[i] = 0.0
                        else:
                            signals[i] = 0.20  # Hold long
                    elif position == -1:  # Short position
                        if c_1h[i] > camarilla_l4:
                            position = 0
                            signals[i] = 0.0
                        else:
                            signals[i] = -0.20  # Hold short
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
        else:
            # Not enough data yet, hold flat
            signals[i] = 0.0
    
    return signals
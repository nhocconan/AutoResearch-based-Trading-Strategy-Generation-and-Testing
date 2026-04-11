#!/usr/bin/env python3
"""
12h_1w_camarilla_breakout_volume_trend_v1
Strategy: 12h Camarilla pivot breakout with volume confirmation and 1w EMA trend filter
Timeframe: 12h
Leverage: 1.0
Hypothesis: Combines Camarilla pivot breakouts from the prior 12h bar with volume confirmation (>2.0x average volume) and filtered by 1w EMA50 trend alignment. Camarilla levels provide precise support/resistance levels that work well in both trending and ranging markets. The 1w EMA filter ensures alignment with the longer-term trend, reducing false breakouts during countertrend moves. Designed for lower frequency (12-37 trades/year) to minimize fee drag while capturing significant moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_camarilla_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous bar
    # For each bar, use previous bar's OHLC to calculate today's levels
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # First bar has no previous data
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Camarilla calculations
    range_val = prev_high - prev_low
    camarilla_h4 = prev_close + (1.1 * range_val / 2)  # Resistance 4
    camarilla_h3 = prev_close + (1.1 * range_val / 4)  # Resistance 3
    camarilla_h2 = prev_close + (1.1 * range_val / 6)  # Resistance 2
    camarilla_h1 = prev_close + (1.1 * range_val / 12) # Resistance 1
    camarilla_l1 = prev_close - (1.1 * range_val / 12) # Support 1
    camarilla_l2 = prev_close - (1.1 * range_val / 6)  # Support 2
    camarilla_l3 = prev_close - (1.1 * range_val / 4)  # Support 3
    camarilla_l4 = prev_close - (1.1 * range_val / 2)  # Support 4
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below 1w EMA50
        uptrend_1w = price_close > ema_50_1w_aligned[i]
        downtrend_1w = price_close < ema_50_1w_aligned[i]
        
        # Breakout conditions using Camarilla levels
        # Long: break above H4 with volume in uptrend
        breakout_up = price_close > camarilla_h4[i]
        # Short: break below L4 with volume in downtrend
        breakout_down = price_close < camarilla_l4[i]
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: upward breakout with volume in uptrend
        long_signal = breakout_up and vol_confirmed and uptrend_1w
        
        # Short: downward breakout with volume in downtrend
        short_signal = breakout_down and vol_confirmed and downtrend_1w
        
        # Exit when price returns to the midpoint (previous close)
        exit_long = position == 1 and price_close < prev_close[i]
        exit_short = position == -1 and price_close > prev_close[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
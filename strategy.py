#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 1w EMA50 trend filter and volume confirmation
# Camarilla R4/S4 levels act as strong intraday support/resistance - breakouts indicate significant momentum shifts
# 1w EMA50 provides long-term trend filter to avoid counter-trend trades in bear markets like 2025
# Volume confirmation (>1.4x average) ensures breakout legitimacy with controlled frequency
# Works in bull/bear: breakouts occur in all regimes, volume confirms legitimacy, weekly trend filter reduces false signals
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and fee drag

name = "4h_Camarilla_R4S4_Breakout_1wEMA50_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Camarilla levels (R4, S4) from previous bar
    # R4 = Close + 1.1*(High-Low)
    # S4 = Close - 1.1*(High-Low)
    hl_range = high - low
    camarilla_r4 = close + 1.1 * hl_range
    camarilla_s4 = close - 1.1 * hl_range
    
    # Need previous bar's levels to avoid look-ahead
    camarilla_r4_prev = np.roll(camarilla_r4, 1)
    camarilla_s4_prev = np.roll(camarilla_s4, 1)
    camarilla_r4_prev[0] = np.nan
    camarilla_s4_prev[0] = np.nan
    
    # Breakout conditions
    breakout_up = close > camarilla_r4_prev
    breakout_down = close < camarilla_s4_prev
    
    # Volume confirmation: volume > 1.4x 20-period average (balanced frequency)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.4 * vol_ma_20)
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r4_prev[i]) or 
            np.isnan(camarilla_s4_prev[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_breakout_up = breakout_up[i]
        curr_breakout_down = breakout_down[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on breakout with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish breakout: price above Camarilla R4 + above 1w EMA50
                if curr_breakout_up and curr_close > curr_ema_50_1w:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price below Camarilla S4 + below 1w EMA50
                elif curr_breakout_down and curr_close < curr_ema_50_1w:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: price closes below Camarilla S4 (reversal) or above Camarilla R4 (take profit)
            if curr_close < camarilla_s4_prev[i] or curr_close > camarilla_r4_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla R4 (reversal) or below Camarilla S4 (take profit)
            if curr_close > camarilla_r4_prev[i] or curr_close < camarilla_s4_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
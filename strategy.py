#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation (>2.0x average)
# Uses tighter volume filter (>2.0x) to reduce trades to target range (75-200 total over 4 years)
# 1d EMA34 provides responsive trend filter for bull/bear markets
# Discrete position sizing: 0.25 for entries to balance return and drawdown
# Works in all regimes: breakouts occur in all markets, volume confirms legitimacy, trend filter avoids counter-trend

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Volume_v2"
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
    
    # Calculate 4h Camarilla levels (R3, S3) from previous bar
    hl_range = high - low
    camarilla_r3 = close + 1.1 * hl_range
    camarilla_s3 = close - 1.1 * hl_range
    
    # Need previous bar's levels to avoid look-ahead
    camarilla_r3_prev = np.roll(camarilla_r3, 1)
    camarilla_s3_prev = np.roll(camarilla_s3, 1)
    camarilla_r3_prev[0] = np.nan
    camarilla_s3_prev[0] = np.nan
    
    # Breakout conditions
    breakout_up = close > camarilla_r3_prev
    breakout_down = close < camarilla_s3_prev
    
    # Volume confirmation: volume > 2.0x 20-period average (tighter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20, 34)  # warmup for Camarilla (20), volume MA (20), EMA (34)
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r3_prev[i]) or 
            np.isnan(camarilla_s3_prev[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_breakout_up = breakout_up[i]
        curr_breakout_down = breakout_down[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on breakout with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish breakout: price above Camarilla R3 + above 1d EMA34
                if curr_breakout_up and curr_close > curr_ema_34_1d:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price below Camarilla S3 + below 1d EMA34
                elif curr_breakout_down and curr_close < curr_ema_34_1d:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: price closes below Camarilla S3 (reversal) or above Camarilla R3 (take profit)
            if curr_close < camarilla_s3_prev[i] or curr_close > camarilla_r3_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla R3 (reversal) or below Camarilla S3 (take profit)
            if curr_close > camarilla_r3_prev[i] or curr_close < camarilla_s3_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
# Hypothesis: 1h strategy using 4h RSI for trend direction and 1d volume spike for confirmation, with 1h RSI for entry timing precision.
# Long when: 4h RSI > 50 (bullish trend), 1d volume > 1.5x 20-period average (volume confirmation), and 1h RSI crosses above 30 from below (oversold bounce).
# Short when: 4h RSI < 50 (bearish trend), 1d volume > 1.5x 20-period average (volume confirmation), and 1h RSI crosses below 70 from above (overbought rejection).
# Exit when 4h RSI crosses back to neutral zone (40-60) or 1h RSI reaches opposite extreme (70 for long exit, 30 for short exit).
# Uses discrete position sizing (0.20) to limit fee churn. Target: 15-37 trades/year by requiring confluence of HTF trend, volume spike, and LTF timing.
# Designed to work in both bull and bear markets: 4h RSI trend filter captures major directional moves, volume confirmation ensures participation,
# and 1h RSI provides precise entries during pullbacks in trending markets.

name = "1h_RSI_Trend_Volume_Timing_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter (RSI)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate RSI(14) on 4h close
    delta_4h = np.diff(close_4h, prepend=close_4h[0])
    gain_4h = np.where(delta_4h > 0, delta_4h, 0)
    loss_4h = np.where(delta_4h < 0, -delta_4h, 0)
    avg_gain_4h = pd.Series(gain_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss_4h = pd.Series(loss_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs_4h = avg_gain_4h / (avg_loss_4h + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs_4h))
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate volume ratio: current volume / 20-period average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / (vol_ma_1d + 1e-10)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # Calculate 1h RSI for entry timing
    delta_1h = np.diff(close, prepend=close[0])
    gain_1h = np.where(delta_1h > 0, delta_1h, 0)
    loss_1h = np.where(delta_1h < 0, -delta_1h, 0)
    avg_gain_1h = pd.Series(gain_1h).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss_1h = pd.Series(loss_1h).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs_1h = avg_gain_1h / (avg_loss_1h + 1e-10)
    rsi_1h = 100 - (100 / (1 + rs_1h))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if np.isnan(rsi_4h_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(rsi_1h[i]):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        long_entry = (rsi_4h_aligned[i] > 50 and 
                      vol_ratio_1d_aligned[i] > 1.5 and 
                      rsi_1h[i] > 30 and rsi_1h[i-1] <= 30)
        
        short_entry = (rsi_4h_aligned[i] < 50 and 
                       vol_ratio_1d_aligned[i] > 1.5 and 
                       rsi_1h[i] < 70 and rsi_1h[i-1] >= 70)
        
        # Exit conditions
        long_exit = (position == 1 and 
                     (rsi_4h_aligned[i] < 40 or rsi_4h_aligned[i] > 60 or 
                      rsi_1h[i] >= 70))
        
        short_exit = (position == -1 and 
                      (rsi_4h_aligned[i] > 40 or rsi_4h_aligned[i] < 60 or 
                       rsi_1h[i] <= 30))
        
        if position == 0:
            if long_entry:
                signals[i] = 0.20
                position = 1
            elif short_entry:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
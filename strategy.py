#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Daily Donchian Breakout with 1w Trend Filter and Volume Confirmation
    # Daily Donchian breakout captures strong momentum, 1w EMA50 filters trend direction,
    # Volume spike confirms institutional participation. Designed for low trade frequency
    # (target: 10-30 trades/year) to survive bear markets with minimal drawdown.
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after Donchian warmup
        # Skip if data not ready
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above upper Donchian + volume spike + price above weekly EMA50
            if close[i] > high_roll[i] and vol_spike[i] and close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian + volume spike + price below weekly EMA50
            elif close[i] < low_roll[i] and vol_spike[i] and close[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses opposite Donchian band or trend reversal vs weekly EMA50
            if position == 1:
                if close[i] < low_roll[i] or close[i] < ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > high_roll[i] or close[i] > ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Donchian_Breakout_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0
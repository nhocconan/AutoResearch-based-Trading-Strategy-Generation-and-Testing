#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla H4/L4 breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above Camarilla H4 level in bull trend (close > 1d EMA50) with volume > 2.0x 20-period MA.
# Short when price breaks below Camarilla L4 level in bear trend (close < 1d EMA50) with volume spike.
# Uses discrete position sizing (0.25) to minimize fee churn. 1d EMA50 provides strong trend filter.
# Volume confirmation ensures institutional participation. Target: 50-150 total trades over 4 years (12-37/year).
# Works in both bull and bear markets: trend filter ensures we only trade in direction of 1d momentum,
# while Camarilla levels provide precise entry/exit points based on intraday price structure.

name = "12h_Camarilla_H4L4_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 1d data for Camarilla levels (based on previous 1d bar's range)
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar
    prev_1d_high = df_1d['high'].values
    prev_1d_low = df_1d['low'].values
    prev_1d_close = df_1d['close'].values
    
    # Calculate Camarilla H4 and L4 levels for each 1d bar
    camarilla_h4 = prev_1d_close + (prev_1d_high - prev_1d_low) * 1.1 / 2
    camarilla_l4 = prev_1d_close - (prev_1d_high - prev_1d_low) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (no additional delay needed as these are based on completed 1d bar)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume regime: current 12h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_1d_aligned[i]
        h4_level = camarilla_h4_aligned[i]
        l4_level = camarilla_l4_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Camarilla breakout conditions
        breakout_h4 = close_val > h4_level
        breakout_l4 = close_val < l4_level
        
        # Entry logic
        if position == 0:
            if is_bull_trend and breakout_h4 and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_bear_trend and breakout_l4 and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below L4 level OR trend reversal
            if close_val < l4_level or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above H4 level OR trend reversal
            if close_val > h4_level or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
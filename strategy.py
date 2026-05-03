#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R4/S4 breakout with 1w EMA50 trend filter and volume confirmation.
# Long when price breaks above Camarilla R4 level in bull trend (close > 1w EMA50) with volume > 2.0x 20-period MA.
# Short when price breaks below Camarilla S4 level in bear trend (close < 1w EMA50) with volume spike.
# Uses discrete position sizing (0.25) to minimize fee churn. 1w EMA50 provides strong trend filter.
# Volume confirmation ensures institutional participation. Target: 30-100 total trades over 4 years (7-25/year).
# Works in both bull and bear markets: trend filter ensures we only trade in direction of 1w momentum,
# while Camarilla levels provide precise entry/exit points based on intraday price structure.

name = "1d_Camarilla_R4S4_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Camarilla levels (based on previous 1d bar's range)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar (R4 and S4)
    prev_1d_high = df_1d['high'].values
    prev_1d_low = df_1d['low'].values
    prev_1d_close = df_1d['close'].values
    
    # Calculate Camarilla R4 and S4 levels for each 1d bar
    camarilla_r4 = prev_1d_close + (prev_1d_high - prev_1d_low) * 1.1 / 2
    camarilla_s4 = prev_1d_close - (prev_1d_high - prev_1d_low) * 1.1 / 2
    
    # Align Camarilla levels to 1d timeframe (no additional delay needed as these are based on completed 1d bar)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume regime: current 1d volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_1w_aligned[i]
        r4_level = camarilla_r4_aligned[i]
        s4_level = camarilla_s4_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Camarilla breakout conditions
        breakout_r4 = close_val > r4_level
        breakout_s4 = close_val < s4_level
        
        # Entry logic
        if position == 0:
            if is_bull_trend and breakout_r4 and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_bear_trend and breakout_s4 and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S4 level OR trend reversal
            if close_val < s4_level or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R4 level OR trend reversal
            if close_val > r4_level or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
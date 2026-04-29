#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA20 trend filter and volume confirmation
# Long when price breaks above Camarilla R1 AND price > 4h EMA20 AND volume > 1.8x 24-bar avg
# Short when price breaks below Camarilla S1 AND price < 4h EMA20 AND volume > 1.8x 24-bar avg
# Exit when price retests Camarilla pivot (central level) or 4h EMA20
# Uses discrete position sizing (0.20) to reduce fee drag. Target: 15-37 trades/year on 1h timeframe.
# Uses session filter (08-20 UTC) to avoid low-liquidity periods. Combines intraday S/R with HTF trend.

name = "1h_Camarilla_R1S1_Breakout_4hEMA20_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid low-liquidity periods
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA20 trend filter and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA20 for trend filter
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate Camarilla levels from previous 4h bar
    # Camarilla: based on previous bar's high, low, close
    # R1 = close + (high-low)*1.1/12
    # S1 = close - (high-low)*1.1/12
    # Pivot = (high+low+close)/3
    prev_high_4h = df_4h['high'].values
    prev_low_4h = df_4h['low'].values
    prev_close_4h = df_4h['close'].values
    
    camarilla_range = prev_high_4h - prev_low_4h
    camarilla_pivot = (prev_high_4h + prev_low_4h + prev_close_4h) / 3.0
    camarilla_r1 = prev_close_4h + camarilla_range * 1.1 / 12.0
    camarilla_s1 = prev_close_4h - camarilla_range * 1.1 / 12.0
    
    # Align Camarilla levels to 1h timeframe (they represent levels from previous 4h bar)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pivot)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Volume confirmation: >1.8x 24-bar average volume (1 day on 1h)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > 1.8 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 20)  # volume MA and EMA20 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(volume_ma_24[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema20_4h = ema_20_4h_aligned[i]
        curr_pivot = camarilla_pivot_aligned[i]
        curr_r1 = camarilla_r1_aligned[i]
        curr_s1 = camarilla_s1_aligned[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price retests Camarilla pivot OR price retests 4h EMA20 (weakening bullish momentum)
            if curr_close <= curr_pivot or curr_close <= curr_ema20_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price retests Camarilla pivot OR price retests 4h EMA20 (weakening bearish momentum)
            if curr_close >= curr_pivot or curr_close >= curr_ema20_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long when price breaks above Camarilla R1 AND price > 4h EMA20 AND volume confirmation
            if curr_close > curr_r1 and curr_close > curr_ema20_4h and vol_conf:
                signals[i] = 0.20
                position = 1
            # Short when price breaks below Camarilla S1 AND price < 4h EMA20 AND volume confirmation
            elif curr_close < curr_s1 and curr_close < curr_ema20_4h and vol_conf:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals
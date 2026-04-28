#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation
# Long when price breaks above R1 AND close > 4h EMA50 AND volume > 2.0x 20-bar avg
# Short when price breaks below S1 AND close < 4h EMA50 AND volume > 2.0x 20-bar avg
# Exits when price retouches the opposite Camarilla level (S1 for longs, R1 for shorts)
# Uses 1h timeframe with 4h/1d HTF filters for direction, targeting 15-37 trades/year.
# Works in bull markets via breakout with trend, works in bear via volume spike filter
# which captures climactic moves preceding reversals.

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_Trend_VolumeFilter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA50 and Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA(50) on 4h close
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels from previous 4h OHLC
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels: R1, S1
    # R1 = close + ((high - low) * 1.1/12)
    # S1 = close - ((high - low) * 1.1/12)
    camarilla_r1 = close_4h + ((high_4h - low_4h) * 1.1 / 12)
    camarilla_s1 = close_4h - ((high_4h - low_4h) * 1.1 / 12)
    
    # Align 4h EMA50 to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Align 4h Camarilla levels to 1h timeframe (use completed 4h bar's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Volume confirmation: >2.0x 20-bar average volume (strict filter to reduce trades)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    # Session filter: 08-20 UTC (reduce noise trades)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        sess_conf = session_filter[i]
        ema_trend = ema_50_4h_aligned[i]
        r1_level = camarilla_r1_aligned[i]
        s1_level = camarilla_s1_aligned[i]
        curr_close = close[i]
        
        # Require both volume and session confirmation
        if not (vol_conf and sess_conf):
            signals[i] = 0.0
            continue
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above R1 AND close > 4h EMA50
            if curr_close > r1_level and curr_close > ema_trend:
                signals[i] = 0.20
                position = 1
            # Short when price breaks below S1 AND close < 4h EMA50
            elif curr_close < s1_level and curr_close < ema_trend:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price retouches S1 (opposite level)
            if curr_close <= s1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short - exit when price retouches R1 (opposite level)
            if curr_close >= r1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
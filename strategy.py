#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above R4 AND close > 12h EMA50 AND volume > 1.8x 24-bar avg
# Short when price breaks below S4 AND close < 12h EMA50 AND volume > 1.8x 24-bar avg
# Exit when price retouches opposite Camarilla level (S4 for longs, R4 for shorts)
# Uses discrete position sizing (0.30) to balance return and fee drag. Target: 20-50 trades/year on 4h.
# Works in bull markets via breakout+trend, works in bear via volume spike requirement
# which captures panic climaxes that often precede reversals. 4h timeframe reduces trade frequency
# while maintaining responsiveness to major moves. Uses 12h HTF for structure and trend.

name = "4h_Camarilla_R4S4_Breakout_12hEMA50_Trend_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 and Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h close
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels from previous 12h OHLC
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels: R4, S4
    # R4 = close + ((high - low) * 1.1/2)
    # S4 = close - ((high - low) * 1.1/2)
    camarilla_r4 = close_12h + ((high_12h - low_12h) * 1.1 / 2)
    camarilla_s4 = close_12h - ((high_12h - low_12h) * 1.1 / 2)
    
    # Align 12h EMA50 to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Align 12h Camarilla levels to 4h timeframe (use completed 12h bar's levels)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # Volume confirmation: >1.8x 24-bar average volume (moderate filter for appropriate trade frequency)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > 1.8 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_trend = ema_50_12h_aligned[i]
        r4_level = camarilla_r4_aligned[i]
        s4_level = camarilla_s4_aligned[i]
        curr_close = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above R4 AND close > 12h EMA50 AND volume confirmation
            if curr_close > r4_level and curr_close > ema_trend and vol_conf:
                signals[i] = 0.30
                position = 1
            # Short when price breaks below S4 AND close < 12h EMA50 AND volume confirmation
            elif curr_close < s4_level and curr_close < ema_trend and vol_conf:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price retouches S4 (opposite level)
            if curr_close <= s4_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:  # Short - exit when price retouches R4 (opposite level)
            if curr_close >= r4_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
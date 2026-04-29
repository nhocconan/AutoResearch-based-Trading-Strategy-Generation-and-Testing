#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike confirmation
# Long when price breaks above Camarilla R1 AND price > 4h EMA50 AND volume > 2.0x 24-bar avg
# Short when price breaks below Camarilla S1 AND price < 4h EMA50 AND volume > 2.0x 24-bar avg
# Exit when price retouches Camarilla pivot point (PP) or opposite breakout occurs
# Uses discrete position sizing (0.20) to minimize fee drag. Target: 15-30 trades/year on 1h.
# Uses 4h EMA50 for trend filter (HTF) and 1d OHLC for Camarilla levels (HTF)
# Session filter: only trade 08:00-20:00 UTC to avoid low-liquidity periods

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_VolumeSpike_v1"
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
    
    # Pre-compute session hours (08:00-20:00 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # Calculate EMA(50) on 4h data
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for Camarilla levels (prior day OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: PP = (H+L+C)/3, R1 = C + (H-L)*1.0/4, S1 = C - (H-L)*1.0/4
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + (high_1d - low_1d) * 1.0 / 4.0
    s1_1d = close_1d - (high_1d - low_1d) * 1.0 / 4.0
    
    # Align Camarilla levels to 1h timeframe (wait for daily bar to close)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume confirmation: >2.0x 24-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > 2.0 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Volume MA needs 24 bars
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(volume_ma_24[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        pp = pp_aligned[i]
        ema_50 = ema_50_4h_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above Camarilla R1 AND price > 4h EMA50 AND volume confirmation
            if curr_high > r1 and curr_close > ema_50 and vol_conf:
                signals[i] = 0.20
                position = 1
            # Short when price breaks below Camarilla S1 AND price < 4h EMA50 AND volume confirmation
            elif curr_low < s1 and curr_close < ema_50 and vol_conf:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price retouches PP or breaks below S1
            if curr_close <= pp or curr_low < s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short - exit when price retouches PP or breaks above R1
            if curr_close >= pp or curr_high > r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
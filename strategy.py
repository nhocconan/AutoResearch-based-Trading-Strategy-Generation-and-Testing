#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Camarilla pivot levels (R1/S1) with 1w EMA50 trend filter and volume confirmation.
# Long when price breaks above 1d Camarilla R1 AND 1w EMA50 > 1w EMA50 previous AND 6h volume > 1.5x 20-period average.
# Short when price breaks below 1d Camarilla S1 AND 1w EMA50 < 1w EMA50 previous AND 6h volume > 1.5x 20-period average.
# Exit when price crosses the 1d Camarilla pivot point (PP).
# Uses discrete position size 0.25. 1d/1w filters provide signal direction, 6h provides entry timing.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.
# This strategy focuses on intraday pivot breaks with higher timeframe trend and volume confirmation,
# which has shown strong performance in DB top performers for SOLUSDT and can work on BTC/ETH.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data once before loop for EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # === 1d Indicators: Camarilla Pivots (based on previous day) ===
    # Camarilla levels calculated from previous day's OHLC
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    # R4 = C + (H - L) * 1.1 / 2
    # S4 = C - (H - L) * 1.1 / 2
    
    # Shift by 1 to use previous day's data for today's levels
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    # Calculate Camarilla levels for each day (using previous day's data)
    camarilla_pp = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    camarilla_r1 = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 12.0
    camarilla_s1 = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 12.0
    camarilla_r4 = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 2.0
    camarilla_s4 = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 2.0
    
    # === 1w Indicators: EMA50 (trend filter) ===
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Trend: EMA50 rising (current > previous)
    ema_50_rising = np.roll(ema_50_1w, 1)  # previous EMA50
    ema_50_rising[0] = np.nan
    ema_50_trend_up = ema_50_1w > ema_50_rising
    ema_50_trend_down = ema_50_1w < ema_50_rising
    
    # Align all indicators to primary timeframe (6h)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    ema_50_trend_up_aligned = align_htf_to_ltf(prices, df_1w, ema_50_trend_up.astype(float))
    ema_50_trend_down_aligned = align_htf_to_ltf(prices, df_1w, ema_50_trend_down.astype(float))
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(ema_50_trend_up_aligned[i]) or 
            np.isnan(ema_50_trend_down_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        pp = camarilla_pp_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        r4 = camarilla_r4_aligned[i]
        s4 = camarilla_s4_aligned[i]
        ema_trend_up = bool(ema_50_trend_up_aligned[i])
        ema_trend_down = bool(ema_50_trend_down_aligned[i])
        price = close[i]
        vol = volume[i]
        
        # Get 6h volume average aligned
        vol_ma_20_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            if price < pp:  # Exit when price crosses below pivot point
                exit_signal = True
        
        elif position == -1:  # Short position
            if price > pp:  # Exit when price crosses above pivot point
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Camarilla R1 AND EMA50 trending up AND volume > 1.5x 20-period avg
            if (price > r1) and ema_trend_up and (vol > 1.5 * vol_ma_20_6h[i]):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Camarilla S1 AND EMA50 trending down AND volume > 1.5x 20-period avg
            elif (price < s1) and ema_trend_down and (vol > 1.5 * vol_ma_20_6h[i]):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_1dCamarillaR1S1_1wEMA50_VolumeConfirmation_V1"
timeframe = "6h"
leverage = 1.0
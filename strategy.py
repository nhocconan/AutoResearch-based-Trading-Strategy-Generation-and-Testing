#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot (R4/S4) breakout with 1w EMA34 trend filter and volume spike (1.8x)
# Long when price breaks above 12h Camarilla R4 AND price > 1w EMA34 (uptrend) AND volume > 1.8x 30-period average
# Short when price breaks below 12h Camarilla S4 AND price < 1w EMA34 (downtrend) AND volume > 1.8x 30-period average
# Exit when price crosses 12h Camarilla midpoint (P) OR 1w EMA34 filter reverses
# Uses Camarilla pivot levels from 12h OHLC + volume confirmation + weekly trend filter to reduce false breakouts
# Designed for 12-37 trades per year over 4 years (50-150 total) to minimize fee drag and maximize edge
# Timeframe: 12h (primary)

name = "12h_Camarilla_R4S4_Breakout_1wEMA34_VolumeSpike_1.8x"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla pivot levels (based on previous bar's OHLC)
    # Camarilla: P = (H+L+C)/3, R4 = C + (H-L)*1.1, S4 = C - (H-L)*1.1
    # We use previous bar's values to avoid look-ahead
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    prev_close_12h = np.roll(close_12h, 1)
    # Set first bar to NaN (no previous bar)
    prev_high_12h[0] = np.nan
    prev_low_12h[0] = np.nan
    prev_close_12h[0] = np.nan
    
    camarilla_p = (prev_high_12h + prev_low_12h + prev_close_12h) / 3.0
    camarilla_r4 = prev_close_12h + (prev_high_12h - prev_low_12h) * 1.1
    camarilla_s4 = prev_close_12h - (prev_high_12h - prev_low_12h) * 1.1
    
    # Get 1w data ONCE before loop for EMA34
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(34)
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 12h timeframe
    camarilla_p_aligned = align_htf_to_ltf(prices, df_12h, camarilla_p)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation on 12h (threshold: 1.8x for tighter filter)
    if len(volume) >= 30:
        vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
        volume_spike = volume > (1.8 * vol_ma_30)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_p_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R4 AND price > EMA34 (uptrend) AND volume spike
            if (close[i] > camarilla_r4_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S4 AND price < EMA34 (downtrend) AND volume spike
            elif (close[i] < camarilla_s4_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Camarilla pivot P OR price < EMA34 (trend weakening)
            if close[i] < camarilla_p_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Camarilla pivot P OR price > EMA34 (trend weakening)
            if close[i] > camarilla_p_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
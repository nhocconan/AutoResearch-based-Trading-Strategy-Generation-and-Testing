#!/usr/bin/env python3
# 4h_Camarilla_R1S1_Breakout_12hTrend_VolumeS
# Hypothesis: Camarilla pivot breakout on 4h with 12h trend filter and volume spike confirmation.
# Long when: 12h uptrend (EMA50 rising), volume > 1.5x 20-period average, and price breaks above Camarilla R1 level.
# Short when: 12h downtrend (EMA50 falling), volume > 1.5x 20-period average, and price breaks below Camarilla S1 level.
# Exit when price returns to Camarilla Pivot level or 12h trend reverses.
# Designed to capture breakouts with institutional levels while avoiding false signals in low-volume or choppy markets.
# Works in bull markets by buying upward breakouts and in bear markets by selling downward breakdowns.
# Camarilla levels provide precise support/resistance, reducing whipsaw vs. generic breakouts.

name = "4h_Camarilla_R1S1_Breakout_12hTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 12h EMA50 for trend ---
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_slope = np.diff(ema_12h, prepend=ema_12h[0])
    ema_12h_uptrend = ema_12h_slope > 0
    ema_12h_downtrend = ema_12h_slope < 0
    
    # Align 12h trend to 4h
    ema_12h_uptrend_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_uptrend)
    ema_12h_downtrend_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_downtrend)
    
    # --- Camarilla levels from previous 12h bar ---
    # Typical price = (H+L+C)/3
    typical_price_12h = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    range_12h = df_12h['high'] - df_12h['low']
    
    # Camarilla levels (based on previous bar)
    R1 = typical_price_12h + (range_12h * 1.0833 / 12)
    S1 = typical_price_12h - (range_12h * 1.0833 / 12)
    PP = typical_price_12h  # Pivot point
    
    # Shift to get previous bar's levels (available at bar open)
    R1_prev = np.roll(R1, 1)
    S1_prev = np.roll(S1, 1)
    PP_prev = np.roll(PP, 1)
    R1_prev[0] = np.nan
    S1_prev[0] = np.nan
    PP_prev[0] = np.nan
    
    # Align Camarilla levels to 4h
    R1_prev_aligned = align_htf_to_ltf(prices, df_12h, R1_prev)
    S1_prev_aligned = align_htf_to_ltf(prices, df_12h, S1_prev)
    PP_prev_aligned = align_htf_to_ltf(prices, df_12h, PP_prev)
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for EMA(50) and volume MA(20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_12h_uptrend_aligned[i]) or
            np.isnan(ema_12h_downtrend_aligned[i]) or
            np.isnan(R1_prev_aligned[i]) or
            np.isnan(S1_prev_aligned[i]) or
            np.isnan(PP_prev_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend from 12h
        is_uptrend = ema_12h_uptrend_aligned[i]
        is_downtrend = ema_12h_downtrend_aligned[i]
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.5  # 50% above average
        
        if position == 0:
            if is_uptrend and vol_spike:
                # Long: 12h uptrend + volume spike + price breaks above Camarilla R1
                if close[i] > R1_prev_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            elif is_downtrend and vol_spike:
                # Short: 12h downtrend + volume spike + price breaks below Camarilla S1
                if close[i] < S1_prev_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: price returns to Camarilla Pivot OR 12h trend turns down
                if close[i] <= PP_prev_aligned[i] or not is_uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to Camarilla Pivot OR 12h trend turns up
                if close[i] >= PP_prev_aligned[i] or not is_downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
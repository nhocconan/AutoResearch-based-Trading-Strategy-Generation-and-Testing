#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Combines Camarilla pivot breakouts from daily timeframe with 1d trend filter and volume confirmation.
# Uses 12h primary timeframe for lower frequency and reduced fee drag. Camarilla R1/S1 levels act as intraday support/resistance.
# In bull markets: buy breakouts above R1 with uptrend confirmation. In bear markets: sell breakdowns below S1 with downtrend confirmation.
# Volume surge filters false breakouts. Trend filter ensures alignment with higher timeframe momentum.
# Designed for 15-35 trades/year to minimize fee drag while capturing meaningful moves.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily OHLC for Camarilla calculation ---
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Camarilla levels (R1, S1) from previous day
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    daily_range = daily_high - daily_low
    camarilla_r1 = daily_close + daily_range * 1.1 / 12
    camarilla_s1 = daily_close - daily_range * 1.1 / 12
    
    # Align Camarilla levels to 12h
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # --- Daily EMA34 for trend filter ---
    ema_34 = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # --- Volume confirmation (2.0x 24-period average) ---
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # --- ATR for stoploss (14-period) ---
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA34 (34) + volume MA (24) + Camarilla (1 day lag)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R1 with uptrend and volume surge
            if close[i] > camarilla_r1_aligned[i] and close[i] > ema_34_aligned[i] and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1 with downtrend and volume surge
            elif close[i] < camarilla_s1_aligned[i] and close[i] < ema_34_aligned[i] and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price drops below Camarilla S1 OR 2.0*ATR trailing stop
                if close[i] < camarilla_s1_aligned[i] or close[i] < high[i] - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises above Camarilla R1 OR 2.0*ATR trailing stop
                if close[i] > camarilla_r1_aligned[i] or close[i] > low[i] + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
# Hypothesis: 12h Williams %R extremes with 1w EMA50 trend filter and 12h volume confirmation (>1.8x 20-period average).
# Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100 over 14 periods.
# Long when Williams %R < -80 (oversold) AND close > 1w EMA50 (bullish trend) AND volume > 1.8x MA20.
# Short when Williams %R > -20 (overbought) AND close < 1w EMA50 (bearish trend) AND volume > 1.8x MA20.
# Exit when Williams %R crosses above -50 (for long) or below -50 (for short) OR price crosses 1w EMA50 in opposite direction.
# Uses 1w HTF for trend to reduce noise and overtrading. Volume confirmation (>1.8x) reduces false signals.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within fee drag limits for 12h timeframe.
# Williams %R is a momentum oscillator that identifies overbought/oversold conditions, effective in ranging markets.

name = "12h_WilliamsR_Extremes_1wEMA50_12hVolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 12h Indicators (LTF) ---
    # Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    # 12h volume confirmation: > 1.8x 20-period average (tight filter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_12h = volume > (1.8 * vol_ma_20)
    
    # --- 1w Indicators (HTF) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # 1w EMA(50) - trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(williams_r[i]) or
            np.isnan(volume_confirm_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND close > 1w EMA50 (bullish trend) AND volume confirm
            if (williams_r[i] < -80 and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_confirm_12h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) AND close < 1w EMA50 (bearish trend) AND volume confirm
            elif (williams_r[i] > -20 and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_confirm_12h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses above -50 (momentum weakening) OR price < 1w EMA50 (trend change)
            if (williams_r[i] > -50 or 
                close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses below -50 (momentum weakening) OR price > 1w EMA50 (trend change)
            if (williams_r[i] < -50 or 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
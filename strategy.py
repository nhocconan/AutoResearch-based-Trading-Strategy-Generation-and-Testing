#!/usr/bin/env python3
# Hypothesis: 6h Williams %R Extreme + 1d EMA34 Trend + Volume Spike (2x MA20)
# Long when Williams %R < -80 (oversold) AND price > 1d EMA34 (bullish trend) AND volume > 2x MA20
# Short when Williams %R > -20 (overbought) AND price < 1d EMA34 (bearish trend) AND volume > 2x MA20
# Exit when Williams %R crosses back above -50 (for long) or below -50 (for short)
# Uses 1d HTF for primary trend to reduce noise and overtrading. Volume spike confirms momentum.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
# Williams %R is a proven mean-reversion oscillator that works in ranging markets, while EMA34 filter ensures we only trade with the higher timeframe trend.

name = "6h_WilliamsR_Extreme_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
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
    
    # Williams %R (14 period) - momentum oscillator
    def williams_r(high, low, close, window=14):
        highest_high = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min().values
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    wr = williams_r(high, low, close, 14)
    
    # Volume confirmation: > 2x 20-period average (volume spike for momentum)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) - trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after Williams %R warmup
        # Skip if missing data
        if (np.isnan(wr[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND price > 1d EMA34 (bullish trend) AND volume spike
            if (wr[i] < -80 and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) AND price < 1d EMA34 (bearish trend) AND volume spike
            elif (wr[i] > -20 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses above -50 (momentum fading)
            if wr[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses below -50 (momentum fading)
            if wr[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
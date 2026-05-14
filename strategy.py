#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and 1d volume spike confirmation.
# Long when price breaks above upper Donchian channel with 1w EMA34 uptrend and 1d volume > 2.0x 20-period average.
# Short when price breaks below lower Donchian channel with 1w EMA34 downtrend and 1d volume > 2.0x 20-period average.
# Exit on opposite Donchian level (lower for longs, upper for shorts).
# Uses discrete position sizing (0.25) to minimize fee churn and strict volume confirmation to reduce false breakouts.
# Target: 30-100 total trades over 4 years = 7-25/year for 1d timeframe.
# Works in bull/bear: 1w EMA34 ensures strong trend alignment, Donchian(20) provides tight structure within trend, volume spike confirms institutional participation.

name = "1d_Donchian20_Breakout_1wEMA34Trend_1dVolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1d Indicators (LTF) ---
    # 1d volume confirmation: > 2.0x 20-period average (stricter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume > (2.0 * vol_ma_20)
    
    # --- 1d Donchian Channel (20-period) ---
    # Upper band: highest high of last 20 periods
    upper_channel = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 periods
    lower_channel = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # --- 1w Indicators (HTF) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # 1w EMA34
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # 1w EMA34 trend: above/below EMA
    ema_uptrend = ema_34_1w_aligned > 0  # Will be handled via close comparison
    ema_downtrend = ema_34_1w_aligned > 0  # Placeholder, will use close vs EMA
    
    # Actually compute trend using close vs EMA (more reliable)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    ema_uptrend = close_1w_aligned > ema_34_1w_aligned
    ema_downtrend = close_1w_aligned < ema_34_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if missing data
        if (np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or
            np.isnan(volume_confirm_1d[i]) or
            np.isnan(ema_uptrend[i]) or
            np.isnan(ema_downtrend[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper Donchian + 1w EMA uptrend + 1d volume spike
            if (close[i] > upper_channel[i] and 
                ema_uptrend[i] and
                volume_confirm_1d[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian + 1w EMA downtrend + 1d volume spike
            elif (close[i] < lower_channel[i] and 
                  ema_downtrend[i] and
                  volume_confirm_1d[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below lower Donchian
            if close[i] < lower_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above upper Donchian
            if close[i] > upper_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
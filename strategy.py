#!/usr/bin/env python3
# Hypothesis: 1d Camarilla R1/S1 breakout with 1w EMA34 trend filter and 1d volume confirmation (>1.5x 20-period average).
# Long when close breaks above R1 AND close > 1w EMA34 (bullish trend) AND volume > 1.5x MA20.
# Short when close breaks below S1 AND close < 1w EMA34 (bearish trend) AND volume > 1.5x MA20.
# Exit when price returns to Camarilla Pivot Point (PP) or crosses 1w EMA34 in opposite direction.
# Uses 1w HTF for trend to reduce noise and overtrading. Volume confirmation (>1.5x) reduces false signals.
# Target: 30-100 total trades over 4 years (7-25/year) to stay within fee drag limits for 1d timeframe.
# Camarilla pivot levels provide intraday support/resistance; breakouts with volume and HTF trend filter capture strong moves.

name = "1d_Camarilla_R1S1_Breakout_1wEMA34_1dVolumeConfirm_v1"
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
    # Camarilla pivot levels (based on previous day's OHLC)
    # PP = (high + low + close) / 3
    # R1 = PP + (high - low) * 1.1 / 12
    # S1 = PP - (high - low) * 1.1 / 12
    # We need previous day's data, so shift by 1
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_high[1] if n > 1 else 0  # fill first value
    prev_low[0] = prev_low[1] if n > 1 else 0
    prev_close[0] = prev_close[1] if n > 1 else 0
    
    pp = (prev_high + prev_low + prev_close) / 3.0
    r1 = pp + (prev_high - prev_low) * 1.1 / 12.0
    s1 = pp - (prev_high - prev_low) * 1.1 / 12.0
    
    # 1d volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume > (1.5 * vol_ma_20)
    
    # --- 1w Indicators (HTF) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # 1w EMA(34) - trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(r1[i]) or
            np.isnan(s1[i]) or
            np.isnan(pp[i]) or
            np.isnan(volume_confirm_1d[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: close breaks above R1 AND close > 1w EMA34 (bullish trend) AND volume confirm
            if (close[i] > r1[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_confirm_1d[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: close breaks below S1 AND close < 1w EMA34 (bearish trend) AND volume confirm
            elif (close[i] < s1[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_confirm_1d[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to PP OR close < 1w EMA34 (trend change)
            if (close[i] <= pp[i] or 
                close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to PP OR close > 1w EMA34 (trend change)
            if (close[i] >= pp[i] or 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
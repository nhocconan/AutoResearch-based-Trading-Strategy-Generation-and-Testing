#!/usr/bin/env python3
# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA50 trend filter and 1d volume spike (>2.0x 20-period average).
# Long when price breaks above R3 AND close > 1w EMA50 (bullish trend) AND volume > 2.0x MA20.
# Short when price breaks below S3 AND close < 1w EMA50 (bearish trend) AND volume > 2.0x MA20.
# Exit when price returns to the Camarilla H4/L4 level (mean reversion) or trend fails.
# Uses 1w HTF for trend to reduce noise and overtrading. Volume confirmation (>2.0x) reduces false signals.
# Target: 30-100 total trades over 4 years (7-25/year) to stay within fee drag limits for 1d timeframe.
# Camarilla pivot levels provide institutional support/resistance; breakouts with volume and HTF trend filter capture strong moves.

name = "1d_Camarilla_R3S3_Breakout_1wEMA50_1dVolumeConfirm_v1"
timeframe = "1d"
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
    
    # --- 1d Indicators (LTF) ---
    # 1d volume confirmation: > 2.0x 20-period average (tight filter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume > (2.0 * vol_ma_20)
    
    # Calculate Camarilla pivot levels for 1d
    # Camarilla: H4 = Close + 1.1*(High-Low)/2, L4 = Close - 1.1*(High-Low)/2
    # R3 = Close + 1.1*(High-Low)/2, S3 = Close - 1.1*(High-Low)/2
    # Actually, Camarilla R3/S3 are: R3 = Close + 1.1*(High-Low)/2, S3 = Close - 1.1*(High-Low)/2
    # H4/L4 are the same as R3/S3 in Camarilla? Let me check: 
    # Standard Camarilla: 
    # H4 = Close + 1.1*(High-Low)/2
    # H3 = Close + 1.1*(High-Low)/4
    # H2 = Close + 1.1*(High-Low)/6
    # H1 = Close + 1.1*(High-Low)/12
    # L1 = Close - 1.1*(High-Low)/12
    # L2 = Close - 1.1*(High-Low)/6
    # L3 = Close - 1.1*(High-Low)/4
    # L4 = Close - 1.1*(High-Low)/2
    # So R3 = H3, S3 = L3? Actually, Camarilla numbers R3/S3 are:
    # R3 = Close + 1.1*(High-Low)/2
    # S3 = Close - 1.1*(High-Low)/2
    # Yes, R3 and S3 are the outer bands.
    camarilla_range = high - low
    r3 = close + 1.1 * camarilla_range / 2
    s3 = close - 1.1 * camarilla_range / 2
    # For exit, we use H4/L4 which are the same as R3/S3? Actually H4/L4 are same as R3/S3.
    # But for mean reversion exit, we can use the close or the pivot point.
    # Let's use the close as the mean reversion level for simplicity.
    # Alternatively, we can use the Camarilla H4/L4 as exit levels.
    # H4 = Close + 1.1*(High-Low)/2 (same as R3)
    # L4 = Close - 1.1*(High-Low)/2 (same as S3)
    # So we'll exit when price returns to the previous day's close (mean reversion).
    # But better to exit when price crosses the Camarilla H3/L3 or returns to the pivot.
    # Let's use the Camarilla H3/L3 for exit: H3 = Close + 1.1*(High-Low)/4, L3 = Close - 1.1*(High-Low)/4
    h3 = close + 1.1 * camarilla_range / 4
    l3 = close - 1.1 * camarilla_range / 4
    
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
            np.isnan(r3[i]) or
            np.isnan(s3[i]) or
            np.isnan(h3[i]) or
            np.isnan(l3[i]) or
            np.isnan(volume_confirm_1d[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above R3 AND close > 1w EMA50 (bullish trend) AND volume confirm
            if (close[i] > r3[i-1] and  # breakout above previous day's R3
                close[i] > ema_50_1w_aligned[i] and 
                volume_confirm_1d[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S3 AND close < 1w EMA50 (bearish trend) AND volume confirm
            elif (close[i] < s3[i-1] and  # breakout below previous day's S3
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_confirm_1d[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to L3 (mean reversion) OR trend fails
            if (close[i] < l3[i] or  # price below Camarilla L3
                close[i] < ema_50_1w_aligned[i]):  # trend failure
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to H3 (mean reversion) OR trend fails
            if (close[i] > h3[i] or  # price above Camarilla H3
                close[i] > ema_50_1w_aligned[i]):  # trend failure
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
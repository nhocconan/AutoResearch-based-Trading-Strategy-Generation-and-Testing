#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and 12h volume confirmation (>1.5x 20-period average).
# Long when price breaks above R3 AND close > 1w EMA50 (bullish trend) AND volume > 1.5x MA20.
# Short when price breaks below S3 AND close < 1w EMA50 (bearish trend) AND volume > 1.5x MA20.
# Exit when price re-enters the Camarilla range (between S3 and R3) OR close crosses 1w EMA50 in opposite direction.
# Uses 1w HTF for trend to reduce noise and overtrading. Volume confirmation (>1.5x) filters weak breakouts.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within fee drag limits for 12h timeframe.
# Camarilla levels provide precise intraday support/resistance; breakouts with volume and HTF trend alignment yield high-probability trades.
# Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend).

name = "12h_Camarilla_R3S3_Breakout_1wEMA50_12hVolumeConfirm_v1"
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
    # Calculate Camarilla levels for current 12h bar using prior 12h bar's OHLC
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We use the *completed* prior 12h bar's OHLC to avoid look-ahead
    # For bar i, we use OHLC from bar i-1 to calculate levels for bar i
    prior_high = np.roll(high, 1)
    prior_low = np.roll(low, 1)
    prior_close = np.roll(close, 1)
    prior_high[0] = prior_low[0] = prior_close[0] = np.nan  # First bar has no prior
    
    rang = prior_high - prior_low
    R3 = prior_close + 1.1 * rang
    S3 = prior_close - 1.1 * rang
    
    # 12h volume confirmation: > 1.5x 20-period average (tight filter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_12h = volume > (1.5 * vol_ma_20)
    
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
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(volume_confirm_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 AND close > 1w EMA50 (bullish trend) AND volume confirm
            if (high[i] > R3[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_confirm_12h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 AND close < 1w EMA50 (bearish trend) AND volume confirm
            elif (low[i] < S3[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_confirm_12h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters Camarilla range (close < R3) OR close < 1w EMA50 (trend change)
            if (close[i] < R3[i] or 
                close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters Camarilla range (close > S3) OR close > 1w EMA50 (trend change)
            if (close[i] > S3[i] or 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
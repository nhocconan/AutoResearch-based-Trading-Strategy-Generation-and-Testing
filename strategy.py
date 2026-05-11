#!/usr/bin/env python3
# 4h_Camarilla_R1S1_Breakout_1dTrend_VolumeS
# Hypothesis: Buy near Camarilla R1 level in uptrend, sell near S1 in downtrend using 1d trend filter and volume confirmation.
# Works in bull markets by buying pullbacks to R1 in uptrends, and in bear markets by selling bounces to S1 in downtrends.
# Uses volume surge to confirm institutional interest and avoids choppy markets with 1d ADX trend filter.

name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d OHLC for Camarilla calculation ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # --- 1d ADX for trend filter (adx > 25 = trending) ---
    # Calculate +DI, -DI, DX
    high_diff = np.diff(high_1d, prepend=high_1d[0])
    low_diff = np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smooth(data, period):
        alpha = 1.0 / period
        result = np.zeros_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    period = 14
    atr = wilders_smooth(tr, period)
    plus_di = 100 * wilders_smooth(plus_dm, period) / (atr + 1e-10)
    minus_di = 100 * wilders_smooth(minus_dm, period) / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = wilders_smooth(dx, period)
    trending = adx > 25  # Strong trend filter
    
    # --- Camarilla levels from previous 1d bar ---
    # Calculate from previous day's OHLC (to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # first bar uses current
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla R1, S1
    R1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    S1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # Align Camarilla levels to 4h
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    trending_aligned = align_htf_to_ltf(prices, df_1d, trending)
    
    # --- 4h EMA21 for dynamic support/resistance ---
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # --- Volume confirmation (volume > 1.5x 20-period EMA) ---
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_surge = volume > 1.5 * vol_ema
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for ADX (14+14) and EMA21
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or
            np.isnan(trending_aligned[i]) or
            np.isnan(ema_21[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            if trending_aligned[i]:
                # Long: Uptrend + price crosses above R1 + volume surge
                if close[i] > R1_aligned[i] and close[i-1] <= R1_aligned[i-1] and vol_surge[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Downtrend + price crosses below S1 + volume surge
                elif close[i] < S1_aligned[i] and close[i-1] >= S1_aligned[i-1] and vol_surge[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: trend ends OR price drops below EMA21
                if not trending_aligned[i] or close[i] < ema_21[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: trend ends OR price rises above EMA21
                if not trending_aligned[i] or close[i] > ema_21[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
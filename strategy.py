#!/usr/bin/env python3
# 4h_ThreeCandleReversal_Pullback
# Hypothesis: Captures mean-reversion pullbacks in strong trends using 3-candle reversal patterns.
# Long when 12h trend up + 3 consecutive lower closes + price near VWAP; short when 12h trend down + 3 consecutive higher closes + price near VWAP.
# Works in bull/bear by trading pullbacks to the mean within the dominant trend, reducing whipsaw vs pure trend following.
# Uses volume confirmation and 1-day ATR for stop management.

name = "4h_ThreeCandleReversal_Pullback"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Get 1d data for ATR and VWAP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 12h EMA34 for trend direction ---
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_slope = ema_34_12h - np.roll(ema_34_12h, 1)
    ema_34_12h_slope[0] = 0
    ema_34_12h_slope = pd.Series(ema_34_12h_slope).ewm(span=3, adjust=False, min_periods=1).mean().values  # smooth slope
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    ema_34_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h_slope)
    
    # --- 1d ATR(14) for volatility ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # --- 4h VWAP (session VWAP reset daily) ---
    # Approximate VWAP using cumulative typical price * volume / cumulative volume
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.where(vwap_den > 0, vwap_num / vwap_den, typical_price)
    
    # --- 3-candle reversal pattern ---
    # For long: 3 consecutive lower closes
    lower_close_1 = close < np.roll(close, 1)
    lower_close_2 = np.roll(close, 1) < np.roll(close, 2)
    lower_close_3 = np.roll(close, 2) < np.roll(close, 3)
    three_lower_close = lower_close_1 & lower_close_2 & lower_close_3
    
    # For short: 3 consecutive higher closes
    higher_close_1 = close > np.roll(close, 1)
    higher_close_2 = np.roll(close, 1) > np.roll(close, 2)
    higher_close_3 = np.roll(close, 2) > np.roll(close, 3)
    three_higher_close = higher_close_1 & higher_close_2 & higher_close_3
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_surge = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for EMA34 (34), ATR (14), and pattern formation
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_12h_aligned[i]) or
            np.isnan(ema_34_12h_slope_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend direction from 12h EMA34 slope
        uptrend = ema_34_12h_slope_aligned[i] > 0
        downtrend = ema_34_12h_slope_aligned[i] < 0
        
        if position == 0:
            if uptrend and vol_surge[i]:
                # Long: 12h uptrend + 3 lower closes + price near VWAP (within 0.5*ATR)
                if three_lower_close[i] and abs(close[i] - vwap[i]) <= 0.5 * atr_14_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            elif downtrend and vol_surge[i]:
                # Short: 12h downtrend + 3 higher closes + price near VWAP (within 0.5*ATR)
                if three_higher_close[i] and abs(close[i] - vwap[i]) <= 0.5 * atr_14_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: 12h trend turns down OR price extends too far from VWAP
                if downtrend or abs(close[i] - vwap[i]) > 1.5 * atr_14_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: 12h trend turns up OR price extends too far from VWAP
                if uptrend or abs(close[i] - vwap[i]) > 1.5 * atr_14_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
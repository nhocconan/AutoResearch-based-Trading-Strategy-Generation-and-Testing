#!/usr/bin/env python3
"""
1d_1w_1wCopper_Cross_Crossover
Hypothesis: Weekly Copper Cross (MACD-style) on 1w with 1d price action confirmation.
Long when weekly MACD crosses above signal line and price > daily VWAP.
Short when weekly MACD crosses below signal line and price < daily VWAP.
Exit on opposite cross or price reversion to VWAP. Uses volume confirmation.
Target: 10-25 trades/year per symbol.
"""

name = "1d_1w_1wCopper_Cross_Crossover"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d VWAP approximation: typical price * volume cumsum / volume cumsum
    typical_price = (high + low + close) / 3.0
    vol_cumsum = np.cumsum(volume)
    tp_vol_cumsum = np.cumsum(typical_price * volume)
    vwap = np.where(vol_cumsum > 0, tp_vol_cumsum / vol_cumsum, typical_price)
    
    # Weekly MACD (12,26,9) - using weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    # EMA12
    ema12 = pd.Series(weekly_close).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA26
    ema26 = pd.Series(weekly_close).ewm(span=26, adjust=False, min_periods=26).mean().values
    # MACD line
    macd_line = ema12 - ema26
    # Signal line (9)
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    # Histogram
    macd_hist = macd_line - signal_line
    
    # Align weekly indicators to daily
    macd_hist_aligned = align_htf_to_ltf(prices, df_1w, macd_hist)
    
    # Detect crosses
    macd_hist_prev = np.roll(macd_hist_aligned, 1)
    macd_hist_prev[0] = 0
    bullish_cross = (macd_hist_aligned > 0) & (macd_hist_prev <= 0)
    bearish_cross = (macd_hist_aligned < 0) & (macd_hist_prev >= 0)
    
    # Volume confirmation: volume > 1.5 * 20-day average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(35, n):
        # Get values
        price = close[i]
        vw = vwap[i]
        bull_cross = bullish_cross[i]
        bear_cross = bearish_cross[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: bullish MACD cross and price above VWAP and volume confirmation
            if bull_cross and price > vw and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: bearish MACD cross and price below VWAP and volume confirmation
            elif bear_cross and price < vw and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: bearish cross or price crosses below VWAP
            if bear_cross or price < vw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: bullish cross or price crosses above VWAP
            if bull_cross or price > vw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
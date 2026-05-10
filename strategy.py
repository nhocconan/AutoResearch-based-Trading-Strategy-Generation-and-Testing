#!/usr/bin/env python3
"""
1h_4h1d_Trend_Reversal_Confirmation
Hypothesis: In 1h timeframe, look for reversals aligned with 4h and 1d trends using EMA crossovers and RSI extremes.
Use 4h EMA20/50 crossover for trend direction and 1d EMA50 for higher timeframe bias.
Enter on 1h RSI(14) < 30 in uptrend or > 70 in downtrend with volume confirmation.
Exit when RSI crosses back to neutral (50) or trend reversals occur.
Session filter (08-20 UTC) to avoid low-liquidity hours.
Target: 20-30 trades/year (80-120 total) to minimize fee drag on 1h timeframe.
Works in bull/bear by using multi-timeframe trend filters to avoid counter-trend trades.
"""

name = "1h_4h1d_Trend_Reversal_Confirmation"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for filtering (08-20 UTC)
    hours = prices.index.hour  # index is already DatetimeIndex
    
    # 4h data for trend direction
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 4h EMA20 and EMA50 for trend
    ema20_4h = np.full(len(close_4h), np.nan)
    ema50_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 50:
        # Initialize EMA20
        ema20_4h[19] = np.mean(close_4h[:20])
        alpha20 = 2 / (20 + 1)
        for i in range(20, len(close_4h)):
            ema20_4h[i] = alpha20 * close_4h[i] + (1 - alpha20) * ema20_4h[i-1]
        # Initialize EMA50
        ema50_4h[49] = np.mean(close_4h[:50])
        alpha50 = 2 / (50 + 1)
        for i in range(50, len(close_4h)):
            ema50_4h[i] = alpha50 * close_4h[i] + (1 - alpha50) * ema50_4h[i-1]
    
    # 1d data for higher timeframe bias
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend bias
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    
    # 1h RSI(14) for entry signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    if n >= 14:
        avg_gain[13] = np.mean(gain[1:14])  # first average of gains
        avg_loss[13] = np.mean(loss[1:14])  # first average of losses
        for i in range(14, n):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align HTF indicators to 1h timeframe
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14)  # Wait for EMA50 and RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        # Trend conditions
        is_4h_uptrend = ema20_4h_aligned[i] > ema50_4h_aligned[i]
        is_4h_downtrend = ema20_4h_aligned[i] < ema50_4h_aligned[i]
        is_1d_uptrend = close[i] > ema50_1d_aligned[i]
        is_1d_downtrend = close[i] < ema50_1d_aligned[i]
        
        # RSI conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_neutral_cross = 40 < rsi[i] < 60  # exit zone
        
        # Volume confirmation: current volume > 1.5x 20-period average
        if i >= 20:
            vol_ma20 = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > 1.5 * vol_ma20
        else:
            volume_confirm = False
        
        if position == 0 and in_session:
            # Long: 4h uptrend, 1d uptrend bias, RSI oversold, volume confirmation
            if (is_4h_uptrend and is_1d_uptrend and rsi_oversold and volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend, 1d downtrend bias, RSI overbought, volume confirmation
            elif (is_4h_downtrend and is_1d_downtrend and rsi_overbought and volume_confirm):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: 4h trend turns down, 1d trend turns down, or RSI returns to neutral
            if (not is_4h_uptrend or not is_1d_uptrend or rsi_neutral_cross):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: 4h trend turns up, 1d trend turns up, or RSI returns to neutral
            if (not is_4h_downtrend or not is_1d_downtrend or rsi_neutral_cross):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
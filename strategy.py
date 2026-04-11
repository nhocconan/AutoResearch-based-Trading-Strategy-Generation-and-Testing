#!/usr/bin/env python3
"""
1d_1w_river_oscillator_volume_v1
Strategy: 1d River Oscillator with volume confirmation and 1w trend filter
Timeframe: 1d
Leverage: 1.0
Hypothesis: Uses daily River Oscillator (34-day RSI of price changes) for mean reversion signals, filtered by weekly trend (price > weekly EMA200) and confirmed by volume spikes (>2x average volume). Designed to capture oversold bounces in bull markets and overbought reversals in bear markets. Target: 20-50 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_river_oscillator_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # River Oscillator: 34-period RSI of daily price changes
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (alpha = 1/period)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[34] = np.mean(gain[1:35])  # First average of gains
    avg_loss[34] = np.mean(loss[1:35])  # First average of losses
    
    for i in range(35, n):
        avg_gain[i] = (avg_gain[i-1] * 33 + gain[i]) / 34
        avg_loss[i] = (avg_loss[i-1] * 33 + loss[i]) / 34
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    river_osc = 100 - (100 / (1 + rs))
    
    # Weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 200:
        ema_200_1w[199] = np.mean(close_1w[:200])
        for i in range(200, len(close_1w)):
            ema_200_1w[i] = (close_1w[i] * 2 + ema_200_1w[i-1] * 198) / 200
    
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume average (20-period)
    vol_avg = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_avg[19] = np.mean(volume[:20])
        for i in range(20, n):
            vol_avg[i] = (volume[i] + vol_avg[i-1] * 19) / 20
    
    vol_spike = volume > (2.0 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if any required data is invalid
        if (np.isnan(river_osc[i]) or np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below weekly EMA200
        uptrend_1w = price_close > ema_200_1w_aligned[i]
        downtrend_1w = price_close < ema_200_1w_aligned[i]
        
        # River Oscillator signals
        oversold = river_osc[i] < 30
        overbought = river_osc[i] > 70
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: oversold with volume in uptrend
        long_signal = oversold and vol_confirmed and uptrend_1w
        
        # Short: overbought with volume in downtrend
        short_signal = overbought and vol_confirmed and downtrend_1w
        
        # Exit when River Oscillator returns to neutral zone (40-60)
        exit_long = position == 1 and river_osc[i] > 40
        exit_short = position == -1 and river_osc[i] < 60
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
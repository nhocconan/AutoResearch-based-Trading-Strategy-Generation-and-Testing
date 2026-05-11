#!/usr/bin/env python3
"""
1d_1w_RSIVolumeBreakout_v1
Hypothesis: Uses weekly RSI extremes (above 70 or below 30) as momentum signals, confirmed by daily volume spikes.
Trades in direction of weekly trend using 200-day EMA filter to avoid counter-trend trades.
Designed for very low trade frequency (<10/year) to minimize fee drag while capturing strong momentum moves.
Works in bull markets (buying dips in uptrends) and bear markets (selling rallies in downtrends).
"""

name = "1d_1w_RSIVolumeBreakout_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 250:  # Need enough data for 200-day EMA
        return np.zeros(n)
    
    # Get weekly data for RSI calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Get daily data for trend filter and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Daily OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- Weekly RSI (14) ---
    close_1w = df_1w['close']
    delta = close_1w.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_values)
    
    # --- Daily 200 EMA for trend filter ---
    close_1d = df_1d['close']
    ema_200_1d = close_1d.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # --- Daily Volume Spike (2x 20-period EMA) ---
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean()
    vol_spike = volume > (2.0 * vol_ema.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 250
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(vol_spike[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine weekly trend
        weekly_uptrend = close[i] > ema_200_1d_aligned[i]
        weekly_downtrend = close[i] < ema_200_1d_aligned[i]
        
        # Entry conditions: RSI extreme + volume spike
        rsi_overbought = rsi_1w_aligned[i] > 70
        rsi_oversold = rsi_1w_aligned[i] < 30
        
        if position == 0:
            if weekly_uptrend and rsi_oversold and vol_spike[i]:
                # Buy dip in uptrend on volume spike
                signals[i] = 0.25
                position = 1
            elif weekly_downtrend and rsi_overbought and vol_spike[i]:
                # Sell rally in downtrend on volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: RSI returns to neutral zone
            if position == 1:
                # Exit long when RSI returns above 40 (bullish momentum fading)
                if rsi_1w_aligned[i] > 40:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short when RSI returns below 60 (bearish momentum fading)
                if rsi_1w_aligned[i] < 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
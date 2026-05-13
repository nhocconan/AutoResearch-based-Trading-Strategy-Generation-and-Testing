#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_RSI
Hypothesis: Camarilla pivot levels (R1/S1) on 4h timeframe act as intraday support/resistance.
Breakouts above R1 or below S1 with volume confirmation, 1d trend alignment (EMA50),
and RSI filter capture momentum moves while avoiding false breakouts. Position size 0.25
limits risk and targets ~25-35 trades/year to minimize fee drag in both bull and bear markets.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_RSI"
timeframe = "4h"
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
    
    # Get 4h data for Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Camarilla levels for each 4h bar
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12, PP = (H+L+C)/3
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    c_4h = df_4h['close'].values
    
    camarilla_pp = (h_4h + l_4h + c_4h) / 3.0
    camarilla_r1 = c_4h + (h_4h - l_4h) * 1.1 / 12.0
    camarilla_s1 = c_4h - (h_4h - l_4h) * 1.1 / 12.0
    
    # Align Camarilla levels to 4h chart (no additional delay needed)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pp)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 2.0x 24-period average (6 days on 4h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    # RSI filter: avoid overbought/oversold extremes
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_filter = (rsi_values > 30) & (rsi_values < 70)  # Avoid extremes
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        if position == 0:
            # LONG: Breakout above R1 with volume confirmation, uptrend, and RSI not overbought
            if (close[i] > camarilla_r1_aligned[i] and 
                volume_filter[i] and 
                close[i] > ema50_1d_aligned[i] and
                rsi_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below S1 with volume confirmation, downtrend, and RSI not oversold
            elif (close[i] < camarilla_s1_aligned[i] and 
                  volume_filter[i] and 
                  close[i] < ema50_1d_aligned[i] and
                  rsi_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot point or trend reverses
            if (close[i] < camarilla_pp_aligned[i]) or \
               (close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to pivot point or trend reverses
            if (close[i] > camarilla_pp_aligned[i]) or \
               (close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
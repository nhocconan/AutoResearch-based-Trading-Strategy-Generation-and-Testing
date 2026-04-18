#!/usr/bin/env python3
"""
6h_12h_RSI_Flush_MeanReversion
Hypothesis: Use RSI extremes on 12h timeframe (RSI < 30 or > 70) combined with 6h price rejection at Bollinger Bands (2σ) to capture mean reversion moves. This works in both bull and bear markets because extreme RSI often precedes reversals, and Bollinger Band rejection provides entry timing. Volume confirmation ensures legitimacy. Targets 15-25 trades/year by requiring alignment of 12h RSI extreme, 6h BB rejection, and volume > 1.3x 20-period average.
"""

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
    
    # Get 12h data for RSI (HTF)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h RSI(14)
    def rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        return 100 - (100 / (1 + rs))
    
    rsi_12h = rsi(close_12h, 14)
    rsi_overbought = rsi_12h > 70
    rsi_oversold = rsi_12h < 30
    
    # Align RSI signals to 6h timeframe
    rsi_overbought_aligned = align_htf_to_ltf(prices, df_12h, rsi_overbought.astype(float))
    rsi_oversold_aligned = align_htf_to_ltf(prices, df_12h, rsi_oversold.astype(float))
    
    # 6h Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = np.full(n, np.nan)
    bb_std_dev = np.full(n, np.nan)
    for i in range(bb_period, n):
        sma[i] = np.mean(close[i-bb_period:i])
        bb_std_dev[i] = np.std(close[i-bb_period:i])
    upper_band = sma + bb_std * bb_std_dev
    lower_band = sma - bb_std * bb_std_dev
    
    # Volume confirmation: current volume > 1.3 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, bb_period)  # need BB and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_overbought_aligned[i]) or np.isnan(rsi_oversold_aligned[i]) or 
            np.isnan(sma[i]) or np.isnan(bb_std_dev[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: 12h RSI oversold, price rejects lower BB, with volume
            if (rsi_oversold_aligned[i] > 0.5 and 
                close[i] < lower_band[i] and 
                close[i] > open_prices[i] and  # bullish rejection: close > open
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: 12h RSI overbought, price rejects upper BB, with volume
            elif (rsi_overbought_aligned[i] > 0.5 and 
                  close[i] > upper_band[i] and 
                  close[i] < open_prices[i] and  # bearish rejection: close < open
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns to SMA (mean reversion complete) or RSI normalizes
            if (close[i] > sma[i] or 
                (not np.isnan(rsi_12h) and len(rsi_12h) > 0 and rsi_12h[-1] > 45)):  # simplified: use last known RSI
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to SMA or RSI normalizes
            if (close[i] < sma[i] or 
                (not np.isnan(rsi_12h) and len(rsi_12h) > 0 and rsi_12h[-1] < 55)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_12h_RSI_Flush_MeanReversion"
timeframe = "6h"
leverage = 1.0
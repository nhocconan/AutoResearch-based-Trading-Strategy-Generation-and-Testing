#!/usr/bin/env python3
"""
Hypothesis: 1d-based strategy using 1-week RSI momentum and volume confirmation to capture medium-term trends.
Long when weekly RSI > 50 and price breaks above daily Donchian(20) upper band with volume confirmation.
Short when weekly RSI < 50 and price breaks below daily Donchian(20) lower band with volume confirmation.
Uses volatility filter (ATR-based) to avoid choppy markets. Targets 7-25 trades/year to minimize fee drag.
Works in both bull and bear markets by following weekly momentum while using daily breakouts for entry timing.
"""
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
    
    # Get 1-week data for RSI(14) momentum filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate RSI(14) on weekly close
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.full(len(close_1w), np.nan)
    avg_loss = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 14:
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        for i in range(14, len(close_1w)):
            avg_gain[i] = (gain[i] + 13 * avg_gain[i-1]) / 14
            avg_loss[i] = (loss[i] + 13 * avg_loss[i-1]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_14_1w = 100 - (100 / (1 + rs))
    
    # Align weekly RSI to daily timeframe
    rsi_14_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_14_1w)
    
    # Calculate daily ATR(14) for volatility filter and position sizing
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[:14])
        else:
            atr[i] = (tr[i] * 1/14) + (atr[i-1] * 13/14)
    
    # Calculate daily Donchian(20) channels
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # need Donchian, ATR, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_14_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Volatility filter: avoid extremely low volatility (choppy) markets
        vol_filter = atr[i] > 0.01 * close[i]  # ATR > 1% of price
        
        # Weekly momentum filter
        bullish_momentum = rsi_14_1w_aligned[i] > 50
        bearish_momentum = rsi_14_1w_aligned[i] < 50
        
        if position == 0:
            # Long entry: price above Donchian upper + weekly bullish momentum + volume + vol filter
            if (close[i] > highest_high[i] and 
                bullish_momentum and 
                vol_confirmed and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short entry: price below Donchian lower + weekly bearish momentum + volume + vol filter
            elif (close[i] < lowest_low[i] and 
                  bearish_momentum and 
                  vol_confirmed and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price below Donchian lower or weekly momentum turns bearish
            if (close[i] < lowest_low[i] or 
                not bullish_momentum):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above Donchian upper or weekly momentum turns bullish
            if (close[i] > highest_high[i] or 
                not bearish_momentum):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyRSI_DonchianBreakout_VolumeFilter"
timeframe = "1d"
leverage = 1.0
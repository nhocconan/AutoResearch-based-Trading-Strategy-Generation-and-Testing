#!/usr/bin/env python3
"""
6h_Price_Action_Actionable
Hypothesis: Price action at daily VWAP with volume confirmation on 6h timeframe.
Enters long when price pulls back to daily VWAP with bullish engulfing candle and volume spike.
Enters short when price rallies to daily VWAP with bearish engulfing candle and volume spike.
Uses daily VWAP as dynamic support/resistance and 6h price action for entry timing.
Designed for low trade frequency (15-35/year) to minimize fee drag in 6BTC/ETH markets.
Works in both bull and bear markets by fading extremes at institutional VWAP levels.
"""

name = "6h_Price_Action_Actionable"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_vwap(high, low, close, volume):
    """Calculate VWAP: cumulative(volume * typical_price) / cumulative(volume)"""
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(volume * typical_price)
    vwap_denominator = np.cumsum(volume)
    vwap = np.where(vwap_denominator != 0, vwap_numerator / vwap_denominator, 0)
    return vwap

def calculate_engulfing(open_price, high, low, close):
    """Calculate bullish and bearish engulfing patterns"""
    bullish_engulf = (close > open_price) & (open_price > np.roll(close, 1)) & (close < np.roll(open_price, 1))
    bearish_engulf = (close < open_price) & (open_price < np.roll(close, 1)) & (close > np.roll(open_price, 1))
    # Handle first element
    bullish_engulf[0] = False
    bearish_engulf[0] = False
    return bullish_engulf, bearish_engulf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for VWAP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    # Calculate daily VWAP
    vwap_1d = calculate_vwap(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, df_1d['volume'].values)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate 6h volume spike (20-period volume ratio)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 0)
    
    # Calculate engulfing patterns
    bullish_engulf, bearish_engulf = calculate_engulfing(open_price, high, low, close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after volume MA warmup
        vwap_val = vwap_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        bull_engulf = bullish_engulf[i]
        bear_engulf = bearish_engulf[i]
        
        # Skip if VWAP is not available
        if np.isnan(vwap_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price at VWAP support + bullish engulfing + volume spike
            if (low[i] <= vwap_val * 1.005 and  # Allow small tolerance for wick
                bull_engulf and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price at VWAP resistance + bearish engulfing + volume spike
            elif (high[i] >= vwap_val * 0.995 and  # Allow small tolerance for wick
                  bear_engulf and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price moves away from VWAP or opposite engulfing
            if (close[i] > vwap_val * 1.02 or bear_engulf):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price moves away from VWAP or opposite engulfing
            if (close[i] < vwap_val * 0.98 or bull_engulf):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
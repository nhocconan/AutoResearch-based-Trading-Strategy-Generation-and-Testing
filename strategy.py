#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1d_engulfing_reversal_v1
# Engulfing candle reversal on 6h timeframe confirmed by 1d trend (EMA21).
# Works in both bull and bear markets by catching reversals at swing points.
# Low trade frequency expected (<30/year) due to strict pattern + trend filter.
name = "6h_1d_engulfing_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (EMA21)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate 1d EMA21 for trend filter
    close_1d = df_1d['close'].values
    ema_21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 1d EMA21 to 6h timeframe
    ema_21_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if EMA not ready
        if np.isnan(ema_21_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Detect bullish engulfing: current green candle fully engulfs previous red candle
        bullish_engulf = (
            close[i] > open_[i] and  # current candle is green
            close[i-1] < open_[i-1] and  # previous candle is red
            close[i] >= open_[i-1] and  # current close >= previous open
            open_[i] <= close[i-1]  # current open <= previous close
        )
        
        # Detect bearish engulfing: current red candle fully engulfs previous green candle
        bearish_engulf = (
            close[i] < open_[i] and  # current candle is red
            close[i-1] > open_[i-1] and  # previous candle is green
            close[i] <= open_[i-1] and  # current close <= previous open
            open_[i] >= close[i-1]  # current open >= previous close
        )
        
        # Trend filter: only take longs in uptrend (price > EMA21), shorts in downtrend (price < EMA21)
        # But allow counter-trend entries at extreme reversals (strong engulfing)
        # Actually, we'll use trend as confirmation: engulfing in direction of trend
        bullish_signal = bullish_engulf and close[i] > ema_21_aligned[i]
        bearish_signal = bearish_engulf and close[i] < ema_21_aligned[i]
        
        # Exit conditions: reverse signal or opposite engulfing
        exit_long = bearish_engulf  # exit long on bearish engulfing
        exit_short = bullish_engulf  # exit short on bullish engulfing
        
        if bullish_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals
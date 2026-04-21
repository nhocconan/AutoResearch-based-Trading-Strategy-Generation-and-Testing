#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1w trend filter and volume confirmation.
Longs when Alligator jaws (13-period smoothed median) are above teeth (8-period smoothed median) 
and lips (5-period smoothed median), with price above jaws and weekly trend up. 
Shorts when jaws below teeth below lips, price below jaws, and weekly trend down.
Exit on Alligator crossover reversal or price crossing jaws.
Designed for 15-25 trades/year on 12h timeframe to minimize fee decay while capturing 
trending moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Williams Alligator components (12h timeframe)
    # Alligator: Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA)
    # SMMA = Smoothed Moving Average (similar to Wilder's smoothing)
    median_price = (prices['high'].values + prices['low'].values) / 2
    
    # Calculate smoothed moving averages (SMMA)
    def smma(series, period):
        result = np.full_like(series, np.nan, dtype=np.float64)
        if len(series) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(series[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(series)):
            result[i] = (result[i-1] * (period-1) + series[i]) / period
        return result
    
    jaw = smma(median_price, 13)  # Jaw (blue)
    teeth = smma(median_price, 8)  # Teeth (red)
    lips = smma(median_price, 5)   # Lips (green)
    
    # Volume confirmation: volume spike > 1.3x 20-period average
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    # ATR for stoploss (14-period)
    tr1 = prices['high'].values - prices['low'].values
    tr2 = np.abs(prices['high'].values - np.roll(prices['close'].values, 1))
    tr3 = np.abs(prices['low'].values - np.roll(prices['close'].values, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        weekly_trend = ema50_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        if position == 0:
            # Enter long: Alligator aligned bullish (Lips > Teeth > Jaw) and price above Jaw
            # with weekly uptrend and volume confirmation
            if (lips_val > teeth_val and teeth_val > jaw_val and 
                price_close > jaw_val and
                weekly_trend > ema50_1w.values[-1] if len(ema50_1w) > 0 else True and  # Simplified trend check
                vol_ratio_val > 1.3):
                signals[i] = 0.25
                position = 1
            # Enter short: Alligator aligned bearish (Lips < Teeth < Jaw) and price below Jaw
            # with weekly downtrend and volume confirmation
            elif (lips_val < teeth_val and teeth_val < jaw_val and 
                  price_close < jaw_val and
                  weekly_trend < ema50_1w.values[-1] if len(ema50_1w) > 0 else True and  # Simplified trend check
                  vol_ratio_val > 1.3):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: Alligator crossover reversal or price crossing Jaw
            exit_signal = False
            
            # Alligator crossover exit
            if position == 1:
                # Exit long if Alligator turns bearish (Lips < Teeth)
                if lips_val < teeth_val:
                    exit_signal = True
                # Exit if price crosses below Jaw
                elif price_close < jaw_val:
                    exit_signal = True
            elif position == -1:
                # Exit short if Alligator turns bullish (Lips > Teeth)
                if lips_val > teeth_val:
                    exit_signal = True
                # Exit if price crosses above Jaw
                elif price_close > jaw_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsAlligator_1wTrend_Volume1.3x_ATR14"
timeframe = "12h"
leverage = 1.0
#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R mean-reversion with 12h EMA trend filter and volume confirmation.
Longs when Williams %R crosses above -80 (oversold) with price above 12h EMA50 and volume>1.3x average.
Shorts when Williams %R crosses below -20 (overbought) with price below 12h EMA50 and volume>1.3x average.
Exit when Williams %R returns to -50 (mean) or 2x ATR stop.
Designed for 20-30 trades/year to minimize fee fade while capturing mean-reversion extremes in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Williams %R (14-period) on 4h data
    high_14 = pd.Series(prices['high'].values).rolling(window=14, min_periods=14).max().values
    low_14 = pd.Series(prices['low'].values).rolling(window=14, min_periods=14).min().values
    close = prices['close'].values
    willr = -100 * (high_14 - close) / (high_14 - low_14)
    
    # Volume confirmation: volume spike > 1.3x 20-period average
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    # ATR for stoploss (20-period)
    tr1 = prices['high'].values - prices['low'].values
    tr2 = np.abs(prices['high'].values - np.roll(prices['close'].values, 1))
    tr3 = np.abs(prices['low'].values - np.roll(prices['close'].values, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(willr[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        willr_val = willr[i]
        ema_trend = ema_50_12h_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        if position == 0:
            # Enter long: Williams %R crosses above -80 (oversold) with uptrend and volume
            if (willr_val > -80 and 
                willr[i-1] <= -80 and 
                price_close > ema_trend and 
                vol_ratio_val > 1.3):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R crosses below -20 (overbought) with downtrend and volume
            elif (willr_val < -20 and 
                  willr[i-1] >= -20 and 
                  price_close < ema_trend and 
                  vol_ratio_val > 1.3):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: Williams %R returns to -50 (mean) OR ATR-based stoploss
            exit_signal = False
            
            # Mean reversion exit: Williams %R crosses -50
            if position == 1 and willr_val < -50:
                exit_signal = True
            elif position == -1 and willr_val > -50:
                exit_signal = True
            
            # ATR-based stoploss (2x ATR from recent extreme)
            if position == 1:
                # For longs, stop below recent low
                recent_low = pd.Series(prices['low'].values).rolling(window=10, min_periods=1).min().iloc[i]
                if price_close < recent_low - 2.0 * atr_val:
                    exit_signal = True
            elif position == -1:
                # For shorts, stop above recent high
                recent_high = pd.Series(prices['high'].values).rolling(window=10, min_periods=1).max().iloc[i]
                if price_close > recent_high + 2.0 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsR_MeanReversion_12hEMA50_Trend_Volume1.3x_ATR2x"
timeframe = "4h"
leverage = 1.0
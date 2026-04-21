#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Fractal reversal with 1d EMA trend filter and volume confirmation.
Longs when bearish fractal forms above EMA34 with volume>1.3x average; shorts when bullish fractal forms below EMA34 with volume>1.3x average.
Exit on opposite fractal formation or 1.5x ATR stop. Designed for 15-25 trades/year to minimize fee drag while capturing high-probability reversals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_hlf

def calculate_williams_fractals(high, low):
    """Calculate Williams fractals: bearish (high) and bullish (low)"""
    n = len(high)
    bearish = np.full(n, np.nan)
    bullish = np.full(n, np.nan)
    
    for i in range(2, n-2):
        # Bearish fractal: high[i] is highest of 5 bars
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish[i] = high[i]
        # Bullish fractal: low[i] is lowest of 5 bars
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            bullish[i] = low[i]
    
    return bearish, bullish

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for fractals and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams fractals on daily
    bearish_fractal, bullish_fractal = calculate_williams_fractals(high_1d, low_1d)
    
    # Calculate EMA34 on daily close
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Williams fractals need 2 extra bars for confirmation (formation + confirmation)
    bearish_fractal_confirmed = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_confirmed = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
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
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(bearish_fractal_confirmed[i]) or np.isnan(bullish_fractal_confirmed[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_val = ema_34_aligned[i]
        bear_fractal = bearish_fractal_confirmed[i]
        bull_fractal = bullish_fractal_confirmed[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        if position == 0:
            # Enter long: bearish fractal above EMA with volume confirmation
            if (not np.isnan(bear_fractal) and 
                bear_fractal > ema_val and 
                vol_ratio_val > 1.3):
                signals[i] = 0.25
                position = 1
            # Enter short: bullish fractal below EMA with volume confirmation
            elif (not np.isnan(bull_fractal) and 
                  bull_fractal < ema_val and 
                  vol_ratio_val > 1.3):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: opposite fractal OR ATR-based stoploss
            exit_signal = False
            
            # Opposite fractal exit
            if position == 1 and not np.isnan(bull_fractal):
                exit_signal = True
            elif position == -1 and not np.isnan(bear_fractal):
                exit_signal = True
            
            # ATR-based stoploss (1.5x ATR from fractal level)
            if position == 1:
                # For longs, stop below bullish fractal level
                if not np.isnan(bull_fractal) and price_close < bull_fractal - 1.5 * atr_val:
                    exit_signal = True
            elif position == -1:
                # For shorts, stop above bearish fractal level
                if not np.isnan(bear_fractal) and price_close > bear_fractal + 1.5 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsFractal_Reversal_1dEMA34_Volume1.3x_ATR1.5x"
timeframe = "4h"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_WilliamsAlligator_ElderRay_Stochastic"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator (13,8,5 SMAs shifted)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Elder Ray: Bull/Bear Power (13-period EMA)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Stochastic (14,3,3)
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    stoch_d = pd.Series(stoch_k).rolling(window=3, min_periods=3).mean().values
    
    # Daily trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: current > 2.0 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or \
           np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or \
           np.isnan(stoch_k[i]) or np.isnan(stoch_d[i]) or np.isnan(ema34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long conditions: Alligator bullish alignment + Elder Ray bullish + Stoch oversold + Daily uptrend + Volume spike
            if (lips[i] > teeth[i] > jaw[i] and  # Alligator bullish alignment
                bull_power[i] > 0 and bear_power[i] < 0 and  # Elder Ray: bulls in control
                stoch_k[i] < 30 and stoch_d[i] < 30 and  # Stoch oversold
                price > ema34_1d_aligned[i] and  # Daily uptrend
                vol_spike[i]):  # Volume confirmation
                signals[i] = 0.25
                position = 1
                continue
            
            # Short conditions: Alligator bearish alignment + Elder Ray bearish + Stoch overbought + Daily downtrend + Volume spike
            elif (lips[i] < teeth[i] < jaw[i] and  # Alligator bearish alignment
                  bear_power[i] < 0 and bull_power[i] > 0 and  # Elder Ray: bears in control
                  stoch_k[i] > 70 and stoch_d[i] > 70 and  # Stoch overbought
                  price < ema34_1d_aligned[i] and  # Daily downtrend
                  vol_spike[i]):  # Volume confirmation
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: Alligator death cross or Elder Ray turns bearish or Stoch overbought
            if (lips[i] < teeth[i] or  # Alligator death cross
                bull_power[i] <= 0 or  # Elder Ray bullish momentum lost
                stoch_k[i] > 70):  # Stoch overbought
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator golden cross or Elder Ray turns bullish or Stoch oversold
            if (lips[i] > teeth[i] or  # Alligator golden cross
                bear_power[i] >= 0 or  # Elder Ray bearish momentum lost
                stoch_k[i] < 30):  # Stoch oversold
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
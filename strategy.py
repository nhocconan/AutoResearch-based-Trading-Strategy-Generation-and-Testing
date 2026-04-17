#!/usr/bin/env python3
"""
4h_RSI_Overbought_Oversold_v1
Mean reversion strategy using RSI(14) with overbought/oversold levels and volume confirmation.
Long when RSI < 30 (oversold) and price near support with volume spike.
Short when RSI > 70 (overbought) and price near resistance with volume spike.
Exit when RSI returns to neutral (40-60) or opposite extreme is reached.
Designed to work in both bull and bear markets by fading extremes.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # === RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === Bollinger Bands for support/resistance context ===
    bb_mid = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1d RSI for higher timeframe filter ===
    df_1d = get_htf_data(prices, '1d')
    delta_1d = np.diff(df_1d['close'].values, prepend=df_1d['close'].values[0])
    gain_1d = np.where(delta_1d > 0, delta_1d, 0)
    loss_1d = np.where(delta_1d < 0, -delta_1d, 0)
    avg_gain_1d = pd.Series(gain_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss_1d = pd.Series(loss_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs_1d = avg_gain_1d / (avg_loss_1d + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs_1d))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmrow = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmrow, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or 
            np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: RSI oversold (<30), price near lower BB, volume confirmed, 1d RSI not extremely oversold
            if (rsi[i] < 30 and 
                close[i] <= bb_lower[i] * 1.02 and  # near or below lower BB
                vol_confirmed and 
                rsi_1d_aligned[i] > 20):  # avoid catching falling knives in strong downtrend
                signals[i] = 0.25
                position = 1
                continue
            # Short: RSI overbought (>70), price near upper BB, volume confirmed, 1d RSI not extremely overbought
            elif (rsi[i] > 70 and 
                  close[i] >= bb_upper[i] * 0.98 and  # near or above upper BB
                  vol_confirmed and 
                  rsi_1d_aligned[i] < 80):  # avoid buying into strong uptrend
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: RSI returns to neutral or reaches opposite extreme
        elif position == 1:
            # Exit long: RSI > 40 (recovering) OR RSI > 70 (overbought extreme)
            if (rsi[i] > 40 or 
                rsi[i] > 70):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI < 60 (declining) OR RSI < 30 (oversold extreme)
            if (rsi[i] < 60 or 
                rsi[i] < 30):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI_Overbought_Oversold_v1"
timeframe = "4h"
leverage = 1.0
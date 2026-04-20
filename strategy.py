#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ElderRay_Energy_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === Daily Elder Ray Components ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 13-period EMA on daily closes
    close_1d_series = pd.Series(close_1d)
    ema13_1d = close_1d_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high_1d - ema13_1d
    # Bear Power = Low - EMA13
    bear_power = low_1d - ema13_1d
    
    # Align to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # === 6h Components ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12-period EMA for trend (2 days)
    close_series = pd.Series(close)
    ema12 = close_series.ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # Volume ratio (15-period average)
    vol_series = pd.Series(volume)
    vol_ma15 = vol_series.rolling(window=15, min_periods=15).mean().values
    vol_ratio = volume / np.where(vol_ma15 > 0, vol_ma15, np.nan)
    
    # RSI(9) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ewm = pd.Series(gain).ewm(alpha=1/9, adjust=False, min_periods=9).mean().values
    loss_ewm = pd.Series(loss).ewm(alpha=1/9, adjust=False, min_periods=9).mean().values
    rs = gain_ewm / np.where(loss_ewm > 0, loss_ewm, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Get values
        close_val = close[i]
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        ema12_val = ema12[i]
        vol_ratio_val = vol_ratio[i]
        rsi_val = rsi[i]
        
        # Skip if any value is NaN
        if (np.isnan(bull_val) or np.isnan(bear_val) or np.isnan(ema12_val) or 
            np.isnan(vol_ratio_val) or np.isnan(rsi_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 (buying pressure) + price above EMA12 + volume confirmation
            if (bull_val > 0 and 
                close_val > ema12_val and 
                vol_ratio_val > 1.3):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (selling pressure) + price below EMA12 + volume confirmation
            elif (bear_val < 0 and 
                  close_val < ema12_val and 
                  vol_ratio_val > 1.3):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bull Power turns negative OR price crosses below EMA12
            if bull_val <= 0 or close_val < ema12_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bear Power turns positive OR price crosses above EMA12
            if bear_val >= 0 or close_val > ema12_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
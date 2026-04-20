#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Elder Ray Index with 1d EMA200 filter and volume confirmation.
# Elder Ray (Bull Power = High - EMA, Bear Power = Low - EMA) measures bull/bear strength.
# Long when Bull Power > 0 and Bear Power rising (momentum), short when Bear Power < 0 and Bull Power falling.
# 1d EMA200 ensures alignment with higher timeframe trend to avoid counter-trend trades.
# Volume filter confirms institutional participation. Target: 20-35 trades/year.

name = "4h_ElderRay_1dEMA200_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for EMA200 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d EMA200 for trend direction ===
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # === Elder Ray on 4h (EMA13) ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # === 4h Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Get values
        close_val = prices['close'].iloc[i]
        ema_val = ema_200_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_val) or np.isnan(bull_val) or np.isnan(bear_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power positive AND rising, price above 1d EMA200, volume confirmation
            if i > 50:
                bull_rising = bull_val > bull_power[i-1]
                if bull_val > 0 and bull_rising and close_val > ema_val and vol_ratio_val > 1.5:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close_val
            # Short: Bear Power negative AND falling, price below 1d EMA200, volume confirmation
            elif i > 50:
                bear_falling = bear_val < bear_power[i-1]
                if bear_val < 0 and bear_falling and close_val < ema_val and vol_ratio_val > 1.5:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close_val
        
        elif position == 1:
            # Long exit: Bull Power turns negative or Bear Power rises above zero
            if bull_val <= 0 or bear_val >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bear Power turns positive or Bull Power falls below zero
            if bear_val >= 0 or bull_val <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
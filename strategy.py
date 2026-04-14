#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d Trend Filter + Volume Confirmation
# Works in bull/bear: Elder Ray measures bull/bear strength relative to EMA, avoiding traps; 1d EMA filters counter-trend trades
# Targets: 12-37 trades/year (50-150 over 4 years) by requiring 3-way confluence
# Position size: 0.25 to manage drawdown

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Elder Ray components (13-period EMA)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(13, n):
        # Get aligned 1d EMA200
        ema_200_i = align_htf_to_ltf(prices, df_1d, ema_200_1d)[i]
        
        if np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(ema_200_i) or np.isnan(vol_ma[i]):
            continue
        
        # Volume confirmation (1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Long: Bull Power > 0 (bulls in control) + price above 1d EMA200 + volume
        if position == 0 and bull_power[i] > 0 and close[i] > ema_200_i and volume_confirm:
            position = 1
            signals[i] = position_size
        # Short: Bear Power < 0 (bears in control) + price below 1d EMA200 + volume
        elif position == 0 and bear_power[i] < 0 and close[i] < ema_200_i and volume_confirm:
            position = -1
            signals[i] = -position_size
        # Exit: Opposite Elder Ray signal or trend reversal
        elif position != 0:
            if position == 1 and bear_power[i] < 0:
                position = 0
                signals[i] = 0.0
            elif position == -1 and bull_power[i] > 0:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_1dEMA200_Volume"
timeframe = "6h"
leverage = 1.0
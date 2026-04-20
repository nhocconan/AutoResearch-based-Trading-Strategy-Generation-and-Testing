#!/usr/bin/env python3
# Strategy: 4h_12h_ChaikinMoneyFlow
# Hypothesis: Chaikin Money Flow (CMF) on 12h measures institutional buying/selling pressure.
# Long when CMF > +0.15 (strong accumulation) and price above 12h EMA50 (uptrend).
# Short when CMF < -0.15 (strong distribution) and price below 12h EMA50 (downtrend).
# Uses volume-weighted accumulation to filter false breakouts. Works in bull/bear by aligning with
# institutional flow. Targets ~30 trades/year via strict CMF thresholds and EMA filter.
# Includes ATR-based stop loss to limit drawdown.
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data for CMF and EMA
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h EMA50 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Chaikin Money Flow (CMF) over 20 periods
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # Money Flow Volume = Money Flow Multiplier * Volume
    # CMF = 20-period sum of Money Flow Volume / 20-period sum of Volume
    high_low = high_12h - low_12h
    high_low[high_low == 0] = 1e-10  # Avoid division by zero
    mfm = ((close_12h - low_12h) - (high_12h - close_12h)) / high_low
    mfv = mfm * volume_12h
    
    # Sum over 20 periods
    mfv_sum = pd.Series(mfv).rolling(window=20, min_periods=20).sum().values
    vol_sum = pd.Series(volume_12h).rolling(window=20, min_periods=20).sum().values
    vol_sum[vol_sum == 0] = 1e-10  # Avoid division by zero
    cmf = mfv_sum / vol_sum
    cmf_aligned = align_htf_to_ltf(prices, df_12h, cmf)
    
    # 4h data for entry timing and stop loss
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # ATR for stop loss (14-period)
    high_low_4h = high_4h - low_4h
    high_close_4h = np.abs(high_4h - np.roll(close_4h, 1))
    low_close_4h = np.abs(low_4h - np.roll(close_4h, 1))
    high_low_4h[0] = high_4h[0] - low_4h[0]
    high_close_4h[0] = np.abs(high_4h[0] - close_4h[0])
    low_close_4h[0] = np.abs(low_4h[0] - close_4h[0])
    tr = np.maximum(high_low_4h, np.maximum(high_close_4h, low_close_4h))
    tr[0] = high_low_4h[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_4h, atr_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(cmf_aligned[i]) or 
            np.isnan(atr_14_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        
        if position == 0:
            # Long: CMF > +0.15 (accumulation) and price above 12h EMA50 (uptrend)
            if (cmf_aligned[i] > 0.15 and 
                price > ema50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: CMF < -0.15 (distribution) and price below 12h EMA50 (downtrend)
            elif (cmf_aligned[i] < -0.15 and 
                  price < ema50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: CMF turns negative or ATR-based stop
            if (cmf_aligned[i] < 0 or 
                price < low_4h[i] - 1.5 * atr_14_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: CMF turns positive or ATR-based stop
            if (cmf_aligned[i] > 0 or 
                price > high_4h[i] + 1.5 * atr_14_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12h_ChaikinMoneyFlow"
timeframe = "4h"
leverage = 1.0
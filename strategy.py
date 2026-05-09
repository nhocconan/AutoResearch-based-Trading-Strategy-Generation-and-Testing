#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy combining weekly (1w) and daily (1d) timeframes.
# Uses 1w for primary trend (EMA50), 1d for entry/exit signals via RSI extremes and price action.
# Enters long when price pulls back to 1w EMA50 during uptrend with RSI < 30 and volume spike.
# Enters short when price rallies to 1w EMA50 during downtrend with RSI > 70 and volume spike.
# Exits on trend reversal or RSI normalization. Designed to work in both bull and bear markets
# by trading pullbacks to the higher timeframe trend. Target: 20-50 trades/year to minimize fee drag.

name = "12h_EMA50_RSI_Pullback"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get 1d data for entry signals (RSI, volume)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate RSI(14) on 1d close for entry signals
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.fillna(50).values  # Fill NaN with 50 (neutral)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume spike filter: current 12h volume > 2.0 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14, 20)  # Need enough data for EMA50 (1w), RSI (14), volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_1w_val = ema50_1w_aligned[i]
        rsi = rsi_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Price near 1w EMA50 (pullback) + uptrend + RSI oversold + volume spike
            if (close[i] <= ema50_1w_val * 1.02 and  # Within 2% above EMA50
                close[i] > ema50_1w_val and          # Above EMA50 (uptrend)
                rsi < 30 and                         # RSI oversold
                vol_spike):                          # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Enter short: Price near 1w EMA50 (pullback) + downtrend + RSI overbought + volume spike
            elif (close[i] >= ema50_1w_val * 0.98 and  # Within 2% below EMA50
                  close[i] < ema50_1w_val and          # Below EMA50 (downtrend)
                  rsi > 70 and                         # RSI overbought
                  vol_spike):                          # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Trend reversal or RSI normalization
            if close[i] <= ema50_1w_val or rsi >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Trend reversal or RSI normalization
            if close[i] >= ema50_1w_val or rsi <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
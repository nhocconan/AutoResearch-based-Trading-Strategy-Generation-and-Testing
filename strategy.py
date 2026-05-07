#!/usr/bin/env python3
# 6h_Volume_Weighted_RSI_Pullback_1wTrend
# Hypothesis: On 6b timeframe, RSI(14) pullbacks to 40-60 range during strong weekly trends
# (above/below weekly EMA50) with volume confirmation (1.5x average) capture high-probability
# mean-reversion entries within the trend. Works in bull markets via long pullbacks in uptrend
# and bear markets via short pullbacks in downtrend. Volume filter ensures institutional
# participation, reducing false signals. Target: 60-120 trades over 4 years (15-30/year).

name = "6h_Volume_Weighted_RSI_Pullback_1wTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # RSI(14) calculation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume ratio: current volume / 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Need 20 for volume MA, 14 for RSI
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # RSI pullback conditions: 40-60 range (avoid extremes)
        rsi_pullback = (rsi_values[i] >= 40) & (rsi_values[i] <= 60)
        
        # Volume confirmation: volume > 1.5x average
        volume_confirm = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: RSI pullback in uptrend with volume
            if rsi_pullback and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: RSI pullback in downtrend with volume
            elif rsi_pullback and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI reaches overbought (>70) or trend reversal
            if rsi_values[i] > 70 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI reaches oversold (<30) or trend reversal
            if rsi_values[i] < 30 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
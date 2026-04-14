#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h ATR-based mean reversion with 1d trend filter and volume confirmation
# Targets: 20-30 trades/year by buying dips in uptrends and selling rallies in downtrends
# Logic: Long when price pulls back to EMA10 in uptrend (price > EMA50) with volume confirmation
#        Short when price rallies to EMA10 in downtrend (price < EMA50) with volume confirmation
#        Exit when price crosses EMA10 in opposite direction or trend weakens
# Position size: 0.25 to manage drawdown

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # EMA10 for entry timing
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # ATR for dynamic thresholds
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(20, n):
        # Get aligned daily EMA50
        ema_50_i = align_htf_to_ltf(prices, df_1d, ema_50_1d)[i]
        
        if np.isnan(ema_10[i]) or np.isnan(ema_50_i) or np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            continue
        
        # Volume confirmation (1.3x average)
        volume_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Dynamic threshold based on ATR
        upper_threshold = ema_10[i] + 0.5 * atr[i]
        lower_threshold = ema_10[i] - 0.5 * atr[i]
        
        # Long: Price pulls back to EMA10 in uptrend with volume confirmation
        if position == 0 and close[i] > ema_50_i and close[i] < upper_threshold and close[i] > ema_10[i] and volume_confirm:
            position = 1
            signals[i] = position_size
        # Short: Price rallies to EMA10 in downtrend with volume confirmation
        elif position == 0 and close[i] < ema_50_i and close[i] > lower_threshold and close[i] < ema_10[i] and volume_confirm:
            position = -1
            signals[i] = -position_size
        # Exit: Price crosses EMA10 in opposite direction or trend weakens
        elif position != 0:
            if position == 1 and (close[i] < ema_10[i] or close[i] < ema_50_i):
                position = 0
                signals[i] = 0.0
            elif position == -1 and (close[i] > ema_10[i] or close[i] > ema_50_i):
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_EMA_Pullback_TrendFilter_Volume"
timeframe = "4h"
leverage = 1.0
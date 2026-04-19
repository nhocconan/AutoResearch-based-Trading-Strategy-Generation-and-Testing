#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-week EMA50 trend filter, 1-day ATR volatility filter,
# and 4-hour RSI mean-reversion with volume confirmation. Weekly EMA50 establishes
# longer-term trend bias, daily ATR sets volatility regime for RSI thresholds,
# and 4h RSI extremes with volume spikes capture mean-reversion opportunities.
# Works in bull/bear by requiring alignment with weekly trend and volatility-adjusted entries.
# Target: 20-40 trades/year per symbol.
name = "4h_EMA50_1w_ATR1d_RSI4h_Volume"
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
    
    # Weekly EMA50 for trend bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily ATR(14) for volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    tr1 = np.maximum(df_1d['high'].values[1:] - df_1d['low'].values[1:],
                     np.abs(df_1d['high'].values[1:] - df_1d['close'].values[:-1]))
    tr2 = np.maximum(np.abs(df_1d['low'].values[1:] - df_1d['close'].values[:-1]),
                     np.zeros(len(tr1)))
    tr = np.concatenate([[np.inf], np.maximum(tr1, tr2)])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 4-hour RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike: volume > 1.8 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Dynamic RSI thresholds based on volatility (ATR)
        # Higher volatility -> wider RSI bands (more extreme readings needed)
        vol_factor = atr_14_1d_aligned[i] / np.mean(atr_14_1d_aligned[max(0, i-50):i+1])
        vol_factor = np.clip(vol_factor, 0.5, 2.0)
        rsi_overbought = 70 + (vol_factor - 1) * 10  # 60-80 range
        rsi_oversold = 30 - (vol_factor - 1) * 10    # 20-40 range
        
        if position == 0:
            # Long: RSI oversold, above weekly EMA50, with volume spike
            if (rsi[i] < rsi_oversold and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought, below weekly EMA50, with volume spike
            elif (rsi[i] > rsi_overbought and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if RSI returns to neutral (50) or below weekly EMA50
            if (rsi[i] >= 50) or (close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if RSI returns to neutral (50) or above weekly EMA50
            if (rsi[i] <= 50) or (close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
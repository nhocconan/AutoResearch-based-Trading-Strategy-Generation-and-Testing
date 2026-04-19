# [Experiment #62879] Hypothesis: 6h timeframe with 12h/1d HTF confluence. Use 12h ADX for trend strength + 1d RSI for mean reversion entries. Long when 12h ADX > 25 (trending) and 1d RSI < 30 (oversold pullback). Short when 12h ADX > 25 and 1d RSI > 70 (overbought pullback). Exit on RSI mean reversion (40-60) or ADX weakening (< 20). This captures trend continuation after pullbacks in both bull/bear markets. Target: 15-30 trades/year.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ADX25_RSI_Pullback_TrendContinuation_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for ADX trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ADX (14-period)
    plus_dm = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    minus_dm = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    tr = np.maximum(high_12h[1:] - low_12h[1:], 
                    np.maximum(np.abs(high_12h[1:] - close_12h[:-1]), 
                               np.abs(low_12h[1:] - close_12h[:-1])))
    tr = np.concatenate([[np.nan], tr])
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_14)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_14)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Get 1d data for RSI mean reversion (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d RSI (14-period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx_12h_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        
        if position == 0:
            # Long: strong trend (ADX>25) + oversold pullback (RSI<30)
            if adx_val > 25 and rsi_val < 30:
                signals[i] = 0.25
                position = 1
            # Short: strong trend (ADX>25) + overbought pullback (RSI>70)
            elif adx_val > 25 and rsi_val > 70:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: RSI mean reversion (40-60) or trend weakening (ADX<20)
            if 40 <= rsi_val <= 60 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: RSI mean reversion (40-60) or trend weakening (ADX<20)
            if 40 <= rsi_val <= 60 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
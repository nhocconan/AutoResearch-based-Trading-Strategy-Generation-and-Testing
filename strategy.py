#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === Weekly EMA(21) for trend filter ===
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # === Daily RSI(14) for mean reversion signal ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === Daily ATR(14) for volatility normalization ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align daily indicators to lower timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # === Volume confirmation (20-period average) ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_21 = ema_21_1w_aligned[i]
        rsi_val = rsi_aligned[i]
        atr_val = atr_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if ATR is too low (avoid choppy markets)
        if atr_val < 0.0001 * price_close:  # Avoid division by near-zero
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Normalized RSI distance from 50 (mean reversion signal)
        rsi_dev = (rsi_val - 50) / (atr_val * 100)  # Scale by volatility
        
        if position == 0:
            # Enter long: RSI oversold in weekly uptrend with volume
            if (rsi_val < 30 and 
                price_close > ema_21 and 
                vol_ratio_val > 1.3):
                signals[i] = 0.25
                position = 1
            # Enter short: RSI overbought in weekly downtrend with volume
            elif (rsi_val > 70 and 
                  price_close < ema_21 and 
                  vol_ratio_val > 1.3):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: RSI returns to neutral or trend reversal
            if position == 1 and (rsi_val > 50 or price_close < ema_21):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (rsi_val < 50 or price_close > ema_21):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WeeklyEMA21_RSI_MeanReversion_Volume"
timeframe = "1d"
leverage = 1.0
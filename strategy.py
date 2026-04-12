#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly RSI(14)
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w = rsi_1w.values
    
    # Calculate weekly ATR(14)
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = np.full(len(df_1w), np.nan)
    for i in range(14, len(df_1w)):
        atr_1w[i] = np.mean(tr[i-14:i+1])
    
    # Calculate weekly volume moving average
    vol_s_1w = pd.Series(volume_1w)
    vol_ma_10_1w = vol_s_1w.rolling(window=10, min_periods=10).mean().values
    
    # Align weekly indicators to daily timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    vol_ma_10_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_10_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(250, n):
        # Skip if data not ready
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(atr_1w_aligned[i]) or 
            np.isnan(vol_ma_10_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.2 * 10-period weekly volume MA
        vol_filter = volume[i] > 1.2 * vol_ma_10_1w_aligned[i]
        
        # Volatility filter: weekly ATR > 0.4 * its 10-period MA
        atr_ma_10_1w = np.full(len(df_1w), np.nan)
        for j in range(23, len(df_1w)):  # 14 + 8 for 10-period MA
            if not np.isnan(np.mean(atr_1w[j-8:j+1])):
                atr_ma_10_1w[j] = np.mean(atr_1w[j-8:j+1])
        atr_ma_10_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_ma_10_1w)
        vol_filter_volatility = (not np.isnan(atr_ma_10_1w_aligned[i]) and 
                                atr_1w_aligned[i] > 0.4 * atr_ma_10_1w_aligned[i])
        
        # RSI conditions for mean reversion
        rsi_oversold = rsi_1w_aligned[i] < 30
        rsi_overbought = rsi_1w_aligned[i] > 70
        
        # Entry conditions: RSI extreme + volume + volatility filter
        long_entry = rsi_oversold and vol_filter and vol_filter_volatility
        short_entry = rsi_overbought and vol_filter and vol_filter_volatility
        
        # Exit conditions: RSI returns to neutral zone
        long_exit = rsi_1w_aligned[i] > 50
        short_exit = rsi_1w_aligned[i] < 50
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_rsi_mean_reversion_vol_filter_v1"
timeframe = "1d"
leverage = 1.0
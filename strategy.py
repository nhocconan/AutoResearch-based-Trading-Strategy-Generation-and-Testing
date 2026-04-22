#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 300:
        return np.zeros(n)
    
    # Load 1d data once for calculations
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) for volatility
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]  # first value
    tr3[0] = tr1[0]  # first value
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 1d RSI(14) for overbought/oversold
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = (100 - (100 / (1 + rs))).values
    
    # Align to 12h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume spike filter (20-period average on 12h data)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(300, n):
        # Skip if any data is not ready
        if (np.isnan(atr_aligned[i]) or 
            np.isnan(ema200_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr = atr_aligned[i]
        ema200 = ema200_aligned[i]
        rsi = rsi_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price > EMA200 + RSI < 30 (oversold) + volume spike
            if price > ema200 and rsi < 30 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price < EMA200 + RSI > 70 (overbought) + volume spike
            elif price < ema200 and rsi > 70 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: ATR-based stop or mean reversion signal
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price drops below EMA200 - 1.5*ATR or RSI > 70
                if price < ema200 - 1.5 * atr or rsi > 70:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price rises above EMA200 + 1.5*ATR or RSI < 30
                if price > ema200 + 1.5 * atr or rsi < 30:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_EMA200_RSI_MeanReversion_Volume"
timeframe = "12h"
leverage = 1.0
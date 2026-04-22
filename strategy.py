#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour mean reversion with 4-hour trend filter and 1-day volatility regime
# Long when: price < 4h VWAP - 0.5*ATR, RSI(14) < 30, and 1d ATR ratio > 1.2 (high vol regime)
# Short when: price > 4h VWAP + 0.5*ATR, RSI(14) > 70, and 1d ATR ratio > 1.2
# Exit when: price crosses 4h VWAP or RSI returns to neutral (40-60)
# Works in both bull and bear markets by fading extremes during high volatility periods
# Uses 4h for directional context and volatility regime, 1h for precise entry/exit
# Target: 20-35 trades/year to stay under fee drag limits

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for VWAP and ATR calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h VWAP (typical price * volume cumulative)
    typical_price_4h = (high_4h + low_4h + close_4h) / 3
    vwap_4h = (typical_price_4h * volume_4h).cumsum() / volume_4h.cumsum()
    
    # Calculate 4h ATR(14)
    tr1_4h = high_4h[1:] - low_4h[1:]
    tr2_4h = np.abs(high_4h[1:] - close_4h[:-1])
    tr3_4h = np.abs(low_4h[1:] - close_4h[:-1])
    tr_4h = np.concatenate([[np.inf], np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))])
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    # Load 1d data for ATR ratio (volatility regime filter)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14)
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.inf], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR ratio: current 1d ATR / 50-period average
    atr_ma_50 = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_1d / atr_ma_50
    
    # Align 4h indicators to 1h timeframe
    vwap_4h_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h)
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # Align 1d ATR ratio to 1h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 1h RSI(14)
    close_1h = prices['close'].values
    delta = np.diff(close_1h, prepend=close_1h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1h = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(vwap_4h_aligned[i]) or 
            np.isnan(atr_4h_aligned[i]) or 
            np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(rsi_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1h[i]
        vwap = vwap_4h_aligned[i]
        atr = atr_4h_aligned[i]
        atr_ratio_val = atr_ratio_aligned[i]
        rsi = rsi_1h[i]
        
        # High volatility regime filter: ATR ratio > 1.2
        high_vol = atr_ratio_val > 1.2
        
        if position == 0:
            # Long conditions: price below VWAP - 0.5*ATR, RSI oversold, high vol
            if price < (vwap - 0.5 * atr) and rsi < 30 and high_vol:
                signals[i] = 0.20
                position = 1
            # Short conditions: price above VWAP + 0.5*ATR, RSI overbought, high vol
            elif price > (vwap + 0.5 * atr) and rsi > 70 and high_vol:
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses VWAP or RSI returns to neutral
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses above VWAP or RSI >= 40
                if price > vwap or rsi >= 40:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses below VWAP or RSI <= 60
                if price < vwap or rsi <= 60:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_VWAP_ATR_RSI_MeanReversion"
timeframe = "1h"
leverage = 1.0
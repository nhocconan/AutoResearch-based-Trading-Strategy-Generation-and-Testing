#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band breakout with 1d RSI filter and volume confirmation.
# Long when price breaks above upper BB(20,2) + RSI > 50 + volume spike.
# Short when price breaks below lower BB(20,2) + RSI < 50 + volume spike.
# Uses Bollinger Bands for volatility-based breakouts and RSI for momentum filter.
# Designed to work in both bull and bear markets by combining volatility breakout with momentum confirmation.
# Bollinger Bands adapt to volatility, making them effective in ranging and trending markets.
name = "4h_BollingerBreakout_1dRSI_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (bb_std * std)
    lower_band = sma - (bb_std * std)
    
    # 1d RSI (14)
    rsi_period = 14
    delta = pd.Series(df_1d['close'].values).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period, min_periods=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period, min_periods=rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_1d = rsi.fillna(50).values  # Fill NaN with 50 for stability
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, rsi_period, 20)
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(sma[i]) or np.isnan(std[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price > upper BB + RSI > 50 + volume spike
            if (price > upper_band[i] and rsi_1d_aligned[i] > 50 and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < lower BB + RSI < 50 + volume spike
            elif (price < lower_band[i] and rsi_1d_aligned[i] < 50 and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < middle BB (SMA) or RSI < 40
            if price < sma[i] or rsi_1d_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > middle BB (SMA) or RSI > 60
            if price > sma[i] or rsi_1d_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
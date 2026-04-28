#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for HTF context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Calculate 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d ATR(14) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d Bollinger Bands for range detection
    bb_period = 20
    bb_std = 2.0
    sma = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (bb_std * bb_std_dev)
    lower_band = sma - (bb_std * bb_std_dev)
    bb_width = (upper_band - lower_band) / sma  # Normalized width
    
    # 1d RSI for momentum confirmation
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean().values
    avg_loss = loss.rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align HTF indicators to 12h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    sma_aligned = align_htf_to_ltf(prices, df_1d, sma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_aligned[i]) or np.isnan(bb_width_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(upper_band_aligned[i]) or 
            np.isnan(lower_band_aligned[i]) or np.isnan(sma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Range filter: only trade when Bollinger Bands are narrow (low volatility)
        range_condition = bb_width_aligned[i] < 0.03  # Tight bands = low volatility
        
        # Momentum filter: RSI not in extreme overbought/oversold
        momentum_condition = (rsi_aligned[i] > 30) and (rsi_aligned[i] < 70)
        
        # Breakout conditions
        long_breakout = close[i] > upper_band_aligned[i]
        short_breakout = close[i] < lower_band_aligned[i]
        
        # Entry conditions: breakout during low volatility + momentum filter
        long_entry = long_breakout and range_condition and momentum_condition
        short_entry = short_breakout and range_condition and momentum_condition
        
        # Exit conditions: return to middle band or volatility expansion
        long_exit = close[i] < sma_aligned[i] or bb_width_aligned[i] > 0.06
        short_exit = close[i] > sma_aligned[i] or bb_width_aligned[i] > 0.06
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_BollingerBreakout_RangeFilter"
timeframe = "12h"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Triple Exponential Moving Average (TEMA) crossover with 1d RSI filter and volume spike confirmation.
# TEMA reduces lag compared to traditional moving averages, providing earlier trend signals.
# A bullish signal occurs when TEMA(9) crosses above TEMA(21) with RSI(14) > 50 on daily timeframe and volume > 1.5x 20-period average.
# Bearish signal when TEMA(9) crosses below TEMA(21) with RSI(14) < 50 and volume spike.
# Designed for low trade frequency (~20-30/year) to minimize fee decay while capturing trends in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for RSI calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate RSI(14) on daily timeframe
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Align 1d RSI to 4h timeframe (waits for 1d bar to close)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # Calculate TEMA(9) and TEMA(21) on 4h close
    close = prices['close'].values
    ema9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean()
    ema9_ema = ema9.ewm(span=9, adjust=False, min_periods=9).mean()
    ema9_ema2 = ema9_ema.ewm(span=9, adjust=False, min_periods=9).mean()
    tema9 = 3 * ema9 - 3 * ema9_ema + ema9_ema2
    
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean()
    ema21_ema = ema21.ewm(span=21, adjust=False, min_periods=21).mean()
    ema21_ema2 = ema21_ema.ewm(span=21, adjust=False, min_periods=21).mean()
    tema21 = 3 * ema21 - 3 * ema21_ema + ema21_ema2
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(tema9[i]) or 
            np.isnan(tema21[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        tema9_val = tema9[i]
        tema21_val = tema21[i]
        rsi_val = rsi_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        # TEMA crossover signals
        tema9_prev = tema9[i-1] if i > 0 else tema9_val
        tema21_prev = tema21[i-1] if i > 0 else tema21_val
        
        bullish_cross = tema9_prev <= tema21_prev and tema9_val > tema21_val
        bearish_cross = tema9_prev >= tema21_prev and tema9_val < tema21_val
        
        if position == 0:
            # Long conditions: bullish TEMA crossover + RSI > 50 + volume spike
            if bullish_cross and rsi_val > 50 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish TEMA crossover + RSI < 50 + volume spike
            elif bearish_cross and rsi_val < 50 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: TEMA cross in opposite direction
            exit_signal = False
            
            if position == 1:  # long position
                if bearish_cross:
                    exit_signal = True
            
            elif position == -1:  # short position
                if bullish_cross:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_TEMA_Crossover_1dRSI_Volume"
timeframe = "4h"
leverage = 1.0
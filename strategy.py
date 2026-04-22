#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1-week RSI mean-reversion and volume confirmation.
# Uses weekly RSI(14) for overbought/oversold signals and 1-day ATR for volatility filtering.
# Designed to capture reversals in both bull and bear markets while avoiding
# choppy conditions. Targets 12-37 trades/year with disciplined risk control.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1-week data for RSI (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly RSI(14)
    close_1w = df_1w['close'].values
    rsi_length = 14
    
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_length, adjust=False, min_periods=rsi_length).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_length, adjust=False, min_periods=rsi_length).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Align RSI to 12h timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Load 1-day data for ATR volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-day ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align ATR to 12h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr = atr_1d_aligned[i]
        rsi = rsi_1w_aligned[i]
        
        # Volatility filter: avoid extremely low volatility environments
        vol_filter = atr > 0.01 * price  # ATR > 1% of price
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long entry: RSI oversold (<30) with volume and volatility
            if rsi < 30 and vol_spike and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: RSI overbought (>70) with volume and volatility
            elif rsi > 70 and vol_spike and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on RSI crossing above 50 (mean reversion)
                if rsi > 50:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on RSI crossing below 50 (mean reversion)
                if rsi < 50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WeeklyRSI_MeanReversion_Volume"
timeframe = "12h"
leverage = 1.0
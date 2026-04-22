#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day timeframe with 1-week RSI mean reversion and volume confirmation.
# Uses weekly RSI (14) to identify extreme overbought/oversold conditions.
# Enters long when weekly RSI < 30 and price > daily VWAP with volume confirmation.
# Enters short when weekly RSI > 70 and price < daily VWAP with volume confirmation.
# Exits when weekly RSI returns to neutral zone (40-60).
# Designed to capture mean reversion moves in both bull and bear markets while avoiding
# choppy conditions. Targets 10-25 trades/year with disciplined risk control.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1-week data for RSI calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly RSI
    close_1w = df_1w['close'].values
    rsi_length = 14
    
    # Calculate price changes
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Calculate average gain and loss
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_length, min_periods=rsi_length, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_length, min_periods=rsi_length, adjust=False).mean().values
    
    # Calculate RSI
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align RSI to daily timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    
    # Load 1-day data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate typical price and VWAP
    typical_price = (high_1d + low_1d + close_1d) / 3
    vwap = pd.Series(typical_price * volume_1d).rolling(window=20, min_periods=20).sum().values / \
           pd.Series(volume_1d).rolling(window=20, min_periods=20).sum().values
    
    # Align VWAP to daily timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(rsi_aligned[i]) or 
            np.isnan(vwap_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        rsi_val = rsi_aligned[i]
        vwap_val = vwap_aligned[i]
        
        # Volume filter: current volume > 1.3 * 20-period average
        vol_filter = vol > 1.3 * vol_ma
        
        if position == 0:
            # Long entry: weekly RSI oversold (<30) and price above VWAP with volume
            if rsi_val < 30 and price > vwap_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: weekly RSI overbought (>70) and price below VWAP with volume
            elif rsi_val > 70 and price < vwap_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: weekly RSI returns to neutral zone (40-60)
            exit_signal = False
            
            if 40 <= rsi_val <= 60:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WeeklyRSI_MeanReversion_VWAP"
timeframe = "1d"
leverage = 1.0
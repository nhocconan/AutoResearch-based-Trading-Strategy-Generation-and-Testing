#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily price action with weekly trend filter, volume confirmation, and RSI mean-reversion.
# Long when price touches weekly VWAP support (below) + RSI < 30 + volume spike.
# Short when price touches weekly VWAP resistance (above) + RSI > 70 + volume spike.
# Exit when price returns to weekly VWAP or RSI reverts to neutral (40-60).
# Designed for low trade frequency (~10-20/year) with edge in ranging markets via mean reversion at institutional levels.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for VWAP and trend context
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly VWAP (volume-weighted average price)
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    vwap_num = np.cumsum(typical_price_1w * volume_1w)
    vwap_den = np.cumsum(volume_1w)
    vwap_1w = vwap_num / vwap_den
    
    # Align weekly VWAP to daily timeframe
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    # Calculate daily RSI(14)
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate daily volume spike (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(vwap_1w_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        vwap = vwap_1w_aligned[i]
        rsi_val = rsi[i]
        
        # Volume filter: current volume > 2.0 * 20-day average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price at or below VWAP support + oversold RSI + volume spike
            if price <= vwap * 1.002 and rsi_val < 30 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price at or above VWAP resistance + overbought RSI + volume spike
            elif price >= vwap * 0.998 and rsi_val > 70 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price returns to VWAP or RSI reverts to neutral
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns above VWAP or RSI exits oversold
                if price > vwap * 1.005 or rsi_val > 40:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns below VWAP or RSI exits overbought
                if price < vwap * 0.995 or rsi_val < 60:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WeeklyVWAP_RSI_MeanReversion_Volume"
timeframe = "1d"
leverage = 1.0
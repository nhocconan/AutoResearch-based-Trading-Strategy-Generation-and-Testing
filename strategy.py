#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion with 4h RSI filter and volume spike.
# Long when price below 1h Bollinger lower band + 4h RSI < 30 + volume spike.
# Short when price above 1h Bollinger upper band + 4h RSI > 70 + volume spike.
# Exit when price crosses back through 1h SMA(20) or volume drops.
# Uses 4h for regime (RSI extremes) and 1h for entry/exit timing.
# Target: 20-30 trades/year to minimize fee drag while capturing mean reversion.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for RSI filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h RSI(14)
    delta = pd.Series(close_4h).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_4h = (100 - (100 / (1 + rs))).values
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # 1h Bollinger Bands (20, 2)
    close = prices['close'].values
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    lower_band = sma_20 - 2 * std_20
    upper_band = sma_20 + 2 * std_20
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(rsi_4h_aligned[i]) or 
            np.isnan(sma_20[i]) or 
            np.isnan(std_20[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        rsi = rsi_4h_aligned[i]
        lower = lower_band[i]
        upper = upper_band[i]
        sma = sma_20[i]
        
        # Volume filter: current volume > 2.0 * 20-day average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price below lower band + RSI < 30 + volume spike
            if price < lower and rsi < 30 and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short conditions: price above upper band + RSI > 70 + volume spike
            elif price > upper and rsi > 70 and vol_spike:
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses back through SMA(20) or volume drops
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses above SMA(20) or volume dries up
                if price > sma or vol < 0.5 * vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses below SMA(20) or volume dries up
                if price < sma or vol < 0.5 * vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Bollinger_RSI_MeanReversion_Volume"
timeframe = "1h"
leverage = 1.0
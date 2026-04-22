#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band breakout with volume spike and 1d RSI filter.
# Long when price breaks above upper band + volume spike + 1d RSI > 50 (bullish regime)
# Short when price breaks below lower band + volume spike + 1d RSI < 50 (bearish regime)
# Exit when price crosses back through middle band or volume drops below 80% of average.
# Works in bull (breakouts with volume) and bear (breakdowns with volume) markets.
# Target: 20-40 trades/year to avoid excessive fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load 1d data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 14-period RSI on 1d
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.values
    
    # Bollinger Bands on 4h (20-period, 2 std dev)
    close = prices['close'].values
    ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std()
    upper = ma_20 + 2 * std_20
    lower = ma_20 - 2 * std_20
    middle = ma_20
    upper = upper.values
    lower = lower.values
    middle = middle.values
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(upper[i]) or 
            np.isnan(lower[i]) or 
            np.isnan(middle[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        # Get 1d RSI aligned (same value for all 4h bars in the day)
        rsi_val = rsi_1d[i // 96] if i // 96 < len(rsi_1d) else rsi_1d[-1]
        
        # Volume filter: current volume > 1.8 * 20-day average
        vol_spike = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above upper band + volume spike + 1d RSI > 50
            if price > upper[i] and vol_spike and rsi_val > 50:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower band + volume spike + 1d RSI < 50
            elif price < lower[i] and vol_spike and rsi_val < 50:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses back through middle band or volume dries up
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below middle band or volume dries up
                if price < middle[i] or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above middle band or volume dries up
                if price > middle[i] or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Bollinger_Breakout_Volume_1dRSI"
timeframe = "4h"
leverage = 1.0
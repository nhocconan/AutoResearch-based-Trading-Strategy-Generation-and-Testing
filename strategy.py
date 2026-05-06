#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12-hour volume-weighted average price (VWAP) deviation with 1-day trend filter
# - Uses 12h VWAP as dynamic support/resistance for mean reversion in ranging markets
# - Uses 1-day EMA50 to determine trend direction (long only above, short only below)
# - Uses 4h RSI(14) for overbought/oversold entry signals
# - Enters long when price is below 12h VWAP AND RSI < 30 AND 1-day EMA50 is rising
# - Enters short when price is above 12h VWAP AND RSI > 70 AND 1-day EMA50 is falling
# - Exits when price returns to 12h VWAP or RSI reaches opposite extreme
# - Designed to capture mean reversion within the prevailing daily trend
# - Target: 60-120 total trades over 4 years (15-30/year) with 0.25 position sizing

name = "4h_12hVWAP_1dEMA50_RSI_MeanReversion"
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
    
    # Get 12h data for VWAP calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # Calculate 12h VWAP (typical price * volume)
    typical_price_12h = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    vwap_12h = (typical_price_12h * df_12h['volume']).cumsum() / df_12h['volume'].cumsum()
    vwap_12h_values = vwap_12h.values
    
    # Align 12h VWAP to 4h timeframe
    vwap_12h_4h = align_htf_to_ltf(prices, df_12h, vwap_12h_values)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA50
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_1d_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1-day EMA50 slope for trend direction
    ema_50_slope = np.zeros_like(ema_50_1d_4h)
    ema_50_slope[1:] = ema_50_1d_4h[1:] - ema_50_1d_4h[:-1]
    
    # RSI calculation (4h timeframe)
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices, prepend=prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        
        avg_gain[period-1] = np.mean(gain[1:period+1])
        avg_loss[period-1] = np.mean(loss[1:period+1])
        
        for i in range(period, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.zeros_like(prices)
        rsi = np.zeros_like(prices)
        for i in range(period, len(prices)):
            if avg_loss[i] != 0:
                rs[i] = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - (100 / (1 + rs[i]))
            else:
                rsi[i] = 100
        
        return rsi
    
    rsi_values = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(vwap_12h_4h[i]) or np.isnan(ema_50_1d_4h[i]) or 
            np.isnan(ema_50_slope[i]) or np.isnan(rsi_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price below VWAP, oversold RSI, and rising 1-day EMA50
            if (close[i] < vwap_12h_4h[i] and 
                rsi_values[i] < 30 and 
                ema_50_slope[i] > 0):
                signals[i] = 0.25
                position = 1
            # Short: price above VWAP, overbought RSI, and falling 1-day EMA50
            elif (close[i] > vwap_12h_4h[i] and 
                  rsi_values[i] > 70 and 
                  ema_50_slope[i] < 0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to VWAP OR RSI reaches overbought
            if close[i] >= vwap_12h_4h[i] or rsi_values[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to VWAP OR RSI reaches oversold
            if close[i] <= vwap_12h_4h[i] or rsi_values[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Wilder's RSI with 14-period and Bollinger Bands for mean reversion in range-bound markets.
# Uses 12h RSI (14) to detect overbought/oversold conditions and Bollinger Bands (20, 2) on 12h for volatility context.
# Long when 12h RSI < 30 and price touches lower Bollinger Band with volume confirmation.
# Short when 12h RSI > 70 and price touches upper Bollinger Band with volume confirmation.
# Exit when RSI crosses back to neutral (40-60 range).
# Designed for low trade frequency (15-25/year) to avoid fee drag. Works in both trending and ranging markets via regime filter.

name = "4h_12hRSI_Bollinger_MeanRev"
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
    
    # Get 12h data for RSI and Bollinger Bands
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h RSI (14-period)
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_12h)
    avg_loss = np.zeros_like(close_12h)
    
    # Wilder's smoothing
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close_12h)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_12h = 100 - (100 / (1 + rs))
    rsi_12h[:13] = np.nan  # Not enough data
    
    # Calculate 12h Bollinger Bands (20, 2)
    sma_20 = np.convolve(close_12h, np.ones(20)/20, mode='same')
    std_20 = np.zeros_like(close_12h)
    
    for i in range(len(close_12h)):
        if i < 19:
            std_20[i] = np.nan
        else:
            std_20[i] = np.std(close_12h[i-19:i+1])
    
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Align 12h indicators to 4h timeframe
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    upper_bb_aligned = align_htf_to_ltf(prices, df_12h, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_12h, lower_bb)
    
    # Volume confirmation: 4h volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for RSI and Bollinger Bands
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_12h_aligned[i]) or 
            np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: RSI < 30 and price at or below lower Bollinger Band with volume confirmation
            if (rsi_12h_aligned[i] < 30 and 
                close[i] <= lower_bb_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: RSI > 70 and price at or above upper Bollinger Band with volume confirmation
            elif (rsi_12h_aligned[i] > 70 and 
                  close[i] >= upper_bb_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI crosses above 40 or price moves above middle of Bollinger Bands
            if rsi_12h_aligned[i] > 40 or close[i] >= sma_20[-1] if len(sma_20) > 0 else False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI crosses below 60 or price moves below middle of Bollinger Bands
            if rsi_12h_aligned[i] < 60 or close[i] <= sma_20[-1] if len(sma_20) > 0 else False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
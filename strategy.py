#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h trading using daily Parabolic SAR for trend direction and 
# weekly momentum confirmation (RSI) to avoid whipsaws. Long when price > SAR 
# and weekly RSI > 50, short when price < SAR and weekly RSI < 50. 
# Uses volume spike for entry confirmation. Designed for low trade frequency 
# (12-37/year) to minimize fee drag while capturing major trends in both 
# bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for RSI (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly RSI (14-period)
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w = np.where(np.isnan(rsi_1w), 50, rsi_1w)
    
    # Align weekly RSI to 12h timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate daily Parabolic SAR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Initialize SAR
    sar = np.zeros_like(close_1d)
    trend = np.ones_like(close_1d)  # 1 for uptrend, -1 for downtrend
    af = 0.02  # acceleration factor
    max_af = 0.2
    ep = 0.0   # extreme point
    
    # Set initial values
    sar[0] = low_1d[0]
    ep = high_1d[0]
    
    for i in range(1, len(close_1d)):
        if trend[i-1] == 1:  # uptrend
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            if low_1d[i] < sar[i]:  # trend reversal
                trend[i] = -1
                sar[i] = ep
                ep = low_1d[i]
                af = 0.02
            else:
                trend[i] = 1
                if high_1d[i] > ep:
                    ep = high_1d[i]
                    af = min(af + 0.02, max_af)
        else:  # downtrend
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            if high_1d[i] > sar[i]:  # trend reversal
                trend[i] = 1
                sar[i] = ep
                ep = high_1d[i]
                af = 0.02
            else:
                trend[i] = -1
                if low_1d[i] < ep:
                    ep = low_1d[i]
                    af = min(af + 0.02, max_af)
    
    # Align daily SAR to 12h timeframe
    sar_aligned = align_htf_to_ltf(prices, df_1d, sar)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(sar_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        sar_val = sar_aligned[i]
        rsi_val = rsi_1w_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long entry: price above SAR, weekly RSI > 50, volume spike
            if price > sar_val and rsi_val > 50 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short entry: price below SAR, weekly RSI < 50, volume spike
            elif price < sar_val and rsi_val < 50 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below SAR
                if price < sar_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above SAR
                if price > sar_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_SAR_WeeklyRSI_Volume"
timeframe = "12h"
leverage = 1.0
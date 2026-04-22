#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Relative Strength Index (RSI) mean reversion on daily timeframe.
# Uses weekly RSI(14) to detect overbought (>70) and oversold (<30) conditions.
# Enters long when weekly RSI crosses below 30 from above (oversold bounce).
# Enters short when weekly RSI crosses above 70 from below (overbought reversal).
# Includes volume confirmation (volume > 1.5x 20-day average) to filter weak signals.
# Designed to work in both bull and bear markets by fading extremes.
# Targets 10-25 trades/year with disciplined risk control.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for RSI calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly RSI(14)
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to alpha=1/14)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)
    rsi = np.where(avg_gain == 0, 0, rsi)
    
    # Align weekly RSI to daily timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    
    # Calculate 20-day average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(rsi_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        rsi_val = rsi_aligned[i]
        rsi_prev = rsi_aligned[i-1] if i > 0 else 50
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long signal: RSI crosses below 30 from above (oversold bounce)
            if rsi_prev > 30 and rsi_val <= 30 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short signal: RSI crosses above 70 from below (overbought reversal)
            elif rsi_prev < 70 and rsi_val >= 70 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when RSI returns to neutral (50) or overbought (70)
                if rsi_val >= 50:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when RSI returns to neutral (50) or oversold (30)
                if rsi_val <= 50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WeeklyRSI_MeanReversion_Volume"
timeframe = "1d"
leverage = 1.0
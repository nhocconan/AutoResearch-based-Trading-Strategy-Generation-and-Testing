#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h 20-period RSI + 1d MACD histogram + volume confirmation
# Long when RSI crosses above 30 (oversold recovery) + MACD bullish (histogram > 0) + volume > 1.5x avg
# Short when RSI crosses below 70 (overbought rejection) + MACD bearish (histogram < 0) + volume > 1.5x avg
# Exit on RSI crossing 50 or opposite extreme (70/30)
# Uses daily MACD for trend filter, RSI for entry timing, volume for confirmation
# Target: 50-150 trades over 4 years with controlled frequency

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d MACD (12,26,9)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # EMAs
    ema12 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = pd.Series(close_1d).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(30, n):
        # Get aligned 1d MACD histogram
        macd_hist_aligned = align_htf_to_ltf(prices, df_1d, macd_hist)[i]
        
        # Check for NaN values
        if (np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or np.isnan(macd_hist_aligned)):
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:  # No position - look for entries
            if volume_confirm:
                # Long: RSI crosses above 30 + MACD bullish
                if rsi[i] > 30 and rsi[i-1] <= 30 and macd_hist_aligned > 0:
                    position = 1
                    signals[i] = position_size
                # Short: RSI crosses below 70 + MACD bearish
                elif rsi[i] < 70 and rsi[i-1] >= 70 and macd_hist_aligned < 0:
                    position = -1
                    signals[i] = -position_size
        elif position == 1:  # Long position - exit on RSI crossing 50 or overbought
            if rsi[i] < 50 or rsi[i] >= 70:
                position = 0
                signals[i] = 0.0
        elif position == -1:  # Short position - exit on RSI crossing 50 or oversold
            if rsi[i] > 50 or rsi[i] <= 30:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_RSI_MACD_Hist_Volume"
timeframe = "12h"
leverage = 1.0
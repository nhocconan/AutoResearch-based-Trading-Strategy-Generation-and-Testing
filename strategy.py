#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy combining 1-day Volume Weighted Average Price (VWAP) as trend filter
# with 4-hour Relative Strength Index (RSI) for mean-reversion entries.
# - Long when price is above 1d VWAP (bullish trend) and 4h RSI < 30 (oversold)
# - Short when price is below 1d VWAP (bearish trend) and 4h RSI > 70 (overbought)
# - Volume confirmation: current volume > 1.5x 20-period average to ensure participation
# - Uses 1d VWAP for robust trend filtering that adapts to intraday volume distribution
# - RSI(14) provides mean-reversion signals within the trend context
# - Target: 80-150 total trades over 4 years (20-38/year) for optimal balance
# - Position size 0.25 for balanced risk exposure

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d VWAP
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    vwap_num = (typical_price_1d * df_1d['volume'].values).cumsum()
    vwap_den = df_1d['volume'].values.cumsum()
    vwap_1d = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    
    # Calculate 4h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(60, n):
        # Skip if any critical data is NaN
        if np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            continue
        
        # Get 1d index for current 4h bar
        # 1d = 6 * 4h bars
        idx_1d = i // 6
        if idx_1d < 1:
            continue
            
        # Previous 1d VWAP (to avoid look-ahead)
        vwap_prev = vwap_1d[idx_1d-1]
        
        # Create arrays for alignment (constant values for the 1d period)
        vwap_arr = np.full(len(df_1d), vwap_prev)
        
        # Align to 4h timeframe
        vwap_4h = align_htf_to_ltf(prices, df_1d, vwap_arr)[i]
        
        if position == 0:
            # Long: price above 1d VWAP + RSI oversold + volume confirmation
            if (close[i] > vwap_4h and  # price above 1d VWAP
                rsi[i] < 30 and  # RSI oversold
                volume[i] > vol_ma[i] * 1.5):  # volume confirmation
                position = 1
                signals[i] = position_size
            # Short: price below 1d VWAP + RSI overbought + volume confirmation
            elif (close[i] < vwap_4h and  # price below 1d VWAP
                  rsi[i] > 70 and  # RSI overbought
                  volume[i] > vol_ma[i] * 1.5):  # volume confirmation
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: RSI overbought or price below 1d VWAP
            if rsi[i] > 70 or close[i] < vwap_4h:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: RSI oversold or price above 1d VWAP
            if rsi[i] < 30 or close[i] > vwap_4h:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_1d_VWAP_RSI_MeanReversion"
timeframe = "4h"
leverage = 1.0
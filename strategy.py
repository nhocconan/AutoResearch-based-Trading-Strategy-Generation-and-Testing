#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyDonchian_RSI_MeanReversion"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 20-week Donchian channels (breakout levels)
    high_20w = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    low_20w = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # Weekly RSI for mean-reversion signal
    rsi_period = 14
    delta = pd.Series(df_1w['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w = rsi_1w.fillna(50).values  # neutral when undefined
    
    # Align weekly indicators to daily
    high_20w_d = align_htf_to_ltf(prices, df_1w, high_20w)
    low_20w_d = align_htf_to_ltf(prices, df_1w, low_20w)
    rsi_1w_d = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 200  # Ensure sufficient history for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(high_20w_d[i]) or np.isnan(low_20w_d[i]) or np.isnan(rsi_1w_d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Mean-reversion logic: buy near weekly support when oversold, sell near weekly resistance when overbought
        if position == 0:
            # Long: price near weekly Donchian low AND weekly RSI oversold (<30)
            if close[i] <= low_20w_d[i] * 1.02 and rsi_1w_d[i] < 30:
                signals[i] = 0.25
                position = 1
            # Short: price near weekly Donchian high AND weekly RSI overbought (>70)
            elif close[i] >= high_20w_d[i] * 0.98 and rsi_1w_d[i] > 70:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price reaches weekly Donchian mid-point or RSI normalizes
            mid_point = (high_20w_d[i] + low_20w_d[i]) / 2
            if close[i] >= mid_point or rsi_1w_d[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches weekly Donchian mid-point or RSI normalizes
            mid_point = (high_20w_d[i] + low_20w_d[i]) / 2
            if close[i] <= mid_point or rsi_1w_d[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
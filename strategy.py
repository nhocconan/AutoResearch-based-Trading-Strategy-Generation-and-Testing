#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data (HTF) once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly RSI (14-period)
    def calculate_rsi(close_prices, period=14):
        delta = np.diff(close_prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close_prices, np.nan)
        avg_loss = np.full_like(close_prices, np.nan)
        
        if len(close_prices) >= period + 1:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
            
            for i in range(period + 1, len(close_prices)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rsi = np.full_like(close_prices, np.nan)
        for i in range(period, len(close_prices)):
            if avg_loss[i] != 0:
                rs = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - (100 / (1 + rs))
            else:
                rsi[i] = 100
        return rsi
    
    rsi_1w = calculate_rsi(close_1w, 14)
    rsi_1d = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate weekly Bollinger Bands (20-period, 2 std)
    def calculate_bollinger(close_prices, period=20, std_dev=2):
        sma = np.full_like(close_prices, np.nan)
        std = np.full_like(close_prices, np.nan)
        upper = np.full_like(close_prices, np.nan)
        lower = np.full_like(close_prices, np.nan)
        
        if len(close_prices) >= period:
            for i in range(period-1, len(close_prices)):
                sma[i] = np.mean(close_prices[i-period+1:i+1])
                std[i] = np.std(close_prices[i-period+1:i+1])
                upper[i] = sma[i] + (std[i] * std_dev)
                lower[i] = sma[i] - (std[i] * std_dev)
        return upper, lower
    
    bb_upper_1w, bb_lower_1w = calculate_bollinger(close_1w, 20, 2)
    bb_upper_1d = align_htf_to_ltf(prices, df_1w, bb_upper_1w)
    bb_lower_1d = align_htf_to_ltf(prices, df_1w, bb_lower_1w)
    
    # Calculate daily ATR (14-period)
    tr = np.zeros(len(prices))
    tr[0] = high[0] - low[0]
    for i in range(1, len(prices)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    atr = np.full(len(prices), np.nan)
    if len(prices) >= 14:
        atr[13] = np.mean(tr[:14])
        for i in range(14, len(prices)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_1d[i]) or
            np.isnan(bb_upper_1d[i]) or
            np.isnan(bb_lower_1d[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Avoid extremely low volatility
        if atr[i] < 0.001 * close[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price touches or breaks below weekly BB lower AND weekly RSI < 30 (oversold)
            if close[i] <= bb_lower_1d[i] and rsi_1d[i] < 30:
                position = 1
                signals[i] = position_size
            # Short: Price touches or breaks above weekly BB upper AND weekly RSI > 70 (overbought)
            elif close[i] >= bb_upper_1d[i] and rsi_1d[i] > 70:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price returns to weekly BB middle (mean reversion complete) OR RSI > 50
            bb_middle = (bb_upper_1d[i] + bb_lower_1d[i]) / 2
            if close[i] >= bb_middle or rsi_1d[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price returns to weekly BB middle OR RSI < 50
            bb_middle = (bb_upper_1d[i] + bb_lower_1d[i]) / 2
            if close[i] <= bb_middle or rsi_1d[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Bollinger_RSI_MeanReversion"
timeframe = "1d"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d RSI(14) filter + volume confirmation.
# Donchian breakout captures trend continuation, 1d RSI avoids overbought/oversold extremes,
# volume confirmation ensures breakout validity. Works in bull/bear markets by
# filtering weak breakouts and choppy conditions. Target: 20-40 trades/year per symbol.
name = "4h_Donchian20_1dRSI14_Volume_Filter"
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
    
    # Get 1d data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate RSI(14) on daily
    def calculate_rsi(close_prices, period=14):
        delta = np.diff(close_prices, prepend=close_prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close_prices)
        avg_loss = np.zeros_like(close_prices)
        
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period+1, len(close_prices)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_14_1d = calculate_rsi(close_1d, 14)
    
    # Donchian(20) channels on 4h
    def calculate_donchian(high_prices, low_prices, period=20):
        upper = np.full_like(high_prices, np.nan)
        lower = np.full_like(low_prices, np.nan)
        
        for i in range(period-1, len(high_prices)):
            upper[i] = np.max(high_prices[i-(period-1):i+1])
            lower[i] = np.min(low_prices[i-(period-1):i+1])
        
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # Align 1d RSI to 4h
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Ensure Donchian and RSI are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or 
            np.isnan(rsi_14_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donch_upper[i]
        lower = donch_lower[i]
        rsi_val = rsi_14_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # RSI filter: avoid extremes (30 < RSI < 70)
        rsi_filter = (rsi_val > 30) and (rsi_val < 70)
        
        if position == 0:
            # Enter long on Donchian upper breakout with volume and RSI filter
            if price > upper and volume_confirmed and rsi_filter:
                signals[i] = 0.25
                position = 1
            # Enter short on Donchian lower breakdown with volume and RSI filter
            elif price < lower and volume_confirmed and rsi_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price crosses below Donchian lower or RSI overbought
            if price < lower or rsi_val >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price crosses above Donchian upper or RSI oversold
            if price > upper or rsi_val <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
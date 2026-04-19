#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d RSI filter and volume confirmation.
# In bull markets: buy breakouts above Donchian(20) high when RSI(14) > 50 (uphill bias).
# In bear markets: sell breakdowns below Donchian(20) low when RSI(14) < 50 (downhill bias).
# Range markets: RSI between 40-60, no trades to avoid whipsaw.
# Volume confirmation: volume > 1.5x 20-period average to filter weak breakouts.
# Target: 20-40 trades/year per symbol to stay within frequency limits.
name = "4h_Donchian_RSI_Volume_Filter"
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
    
    # Get daily data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate RSI (14-period) on daily
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices, prepend=prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        
        if len(prices) > period:
            avg_gain[period] = np.mean(gain[1:period+1])
            avg_loss[period] = np.mean(loss[1:period+1])
            
            for i in range(period+1, len(prices)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1d = calculate_rsi(close_1d, 14)
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian channels (20-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donch_high = rolling_max(high_4h, 20)
    donch_low = rolling_min(low_4h, 20)
    
    # Align indicators to 4h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # Get 4h average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure RSI (14*2+6), Donchian (20), and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi_aligned[i]
        donch_high_val = donch_high_aligned[i]
        donch_low_val = donch_low_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: price breaks above Donchian high AND RSI > 50 (bullish bias)
            if price > donch_high_val and rsi_val > 50 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND RSI < 50 (bearish bias)
            elif price < donch_low_val and rsi_val < 50 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low OR RSI < 40
            if price < donch_low_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high OR RSI > 60
            if price > donch_high_val or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
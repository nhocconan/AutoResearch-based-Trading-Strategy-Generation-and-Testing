#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with weekly RSI filter and volume confirmation.
# Use weekly RSI to determine trend strength: RSI > 60 = bullish, RSI < 40 = bearish, neutral zone = no trades.
# In bullish regime: long when price breaks above Donchian(20) high.
# In bearish regime: short when price breaks below Donchian(20) low.
# Volume confirmation: volume > 1.5x 20-period average.
# Target: 15-30 trades/year per symbol to stay within frequency limits.
name = "1d_Donchian_RSI_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for RSI calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate RSI (14-period)
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    avg_gain = wilder_smooth(gain, 14)
    avg_loss = wilder_smooth(loss, 14)
    rs = np.where(avg_loss == 0, 0, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # Get daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
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
    
    donch_high = rolling_max(high_1d, 20)
    donch_low = rolling_min(low_1d, 20)
    
    # Align indicators to daily timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Get daily average volume for confirmation
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
        
        # Regime determination based on weekly RSI
        is_bullish = rsi_val > 60
        is_bearish = rsi_val < 40
        # Neutral zone (40-60) - no trades
        
        if position == 0:
            # Determine entry based on regime
            if is_bullish and volume_confirmed:
                # Bullish regime: long breakout
                if price > donch_high_val:
                    signals[i] = 0.25
                    position = 1
            elif is_bearish and volume_confirmed:
                # Bearish regime: short breakdown
                if price < donch_low_val:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low
            if price < donch_low_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high
            if price > donch_high_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
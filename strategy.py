#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian(20) breakout with 1h volume confirmation and ATR-based stoploss
# Long when price breaks above 1d Donchian upper channel AND 1h volume > 1.5 * avg_volume(20) AND ATR(14) < 0.03 * price
# Short when price breaks below 1d Donchian lower channel AND 1h volume > 1.5 * avg_volume(20) AND ATR(14) < 0.03 * price
# Exit when price crosses 1d Donchian midpoint (mean reversion to equilibrium)
# Uses discrete sizing 0.25 to balance return and drawdown
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# 1d Donchian provides daily structure with proven breakout edge
# Volume confirmation reduces false breakouts
# Low ATR filter avoids high volatility choppy markets

name = "4h_1dDonchian20_1hVolumeSpike_LowATR_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need sufficient data for Donchian
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        middle = (upper + lower) / 2.0
        return upper, lower, middle
    
    donchian_upper_1d, donchian_lower_1d, donchian_middle_1d = calculate_donchian(high_1d, low_1d, period=20)
    
    # Get 1h data ONCE before loop for volume and ATR
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 30:  # Need sufficient data for volume average and ATR
        return np.zeros(n)
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    volume_1h = df_1h['volume'].values
    
    # Calculate 1h volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20_1h = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    volume_spike_1h = volume_1h > (1.5 * avg_volume_20_1h)
    
    # Calculate 1h ATR(14) for volatility filter
    def calculate_atr(high, low, close, period=14):
        tr1 = np.abs(high - low)
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period TR is high-low
        
        atr = np.zeros_like(high)
        atr[:period-1] = np.nan
        if len(high) >= period:
            atr[period-1] = np.mean(tr[:period])
            for i in range(period, len(high)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_1h = calculate_atr(high_1h, low_1h, close_1h, period=14)
    atr_percent_1h = atr_1h / close_1h  # ATR as percentage of price
    low_atr_filter_1h = atr_percent_1h < 0.03  # Avoid high volatility (>3% ATR)
    
    # Align 1d Donchian levels to 4h timeframe (wait for completed 1d bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_1d)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_1d)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1d, donchian_middle_1d)
    
    # Align 1h indicators to 4h timeframe (wait for completed 1h bar)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1h, volume_spike_1h)
    low_atr_filter_aligned = align_htf_to_ltf(prices, df_1h, low_atr_filter_1h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(low_atr_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper with volume spike and low ATR
            if (close[i] > donchian_upper_aligned[i] and close[i-1] <= donchian_upper_aligned[i-1] and 
                volume_spike_aligned[i] and low_atr_filter_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian lower with volume spike and low ATR
            elif (close[i] < donchian_lower_aligned[i] and close[i-1] >= donchian_lower_aligned[i-1] and 
                  volume_spike_aligned[i] and low_atr_filter_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1d Donchian midpoint (mean reversion)
            if close[i] < donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1d Donchian midpoint (mean reversion)
            if close[i] > donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
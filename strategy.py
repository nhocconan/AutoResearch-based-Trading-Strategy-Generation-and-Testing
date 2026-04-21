#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(15) Breakout + 1w RSI(14) Filter + 1d Volume Spike
# Long when price breaks above Donchian(15) high and 1w RSI > 50 and 1d volume > 2x 20-period average
# Short when price breaks below Donchian(15) low and 1w RSI < 50 and 1d volume > 2x 20-period average
# Exit when price crosses Donchian midpoint
# Weekly RSI filters for market regime (bullish/bearish)
# Volume spike confirms breakout strength
# Target: 20-35 trades/year by requiring RSI filter + volume spike + Donchian breakout

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w and 1d data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1w RSI(14)
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    rsi_period = 14
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    
    avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
    avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
    
    for i in range(rsi_period+1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w[:rsi_period+1] = np.nan
    
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate Donchian(15) on 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    donchian_high = pd.Series(high).rolling(window=15, min_periods=15).max().values
    donchian_low = pd.Series(low).rolling(window=15, min_periods=15).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(15, n):
        # Skip if data not ready
        if np.isnan(rsi_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price
        price = prices['close'].iloc[i]
        
        # Get 1d volume for current bar (approximate)
        # For 6b data, we approximate 1d volume as the current day's volume
        volume = df_1d['volume'].iloc[min(i // 4, len(df_1d)-1)] if i >= 4 else df_1d['volume'].iloc[0]
        
        # Volume confirmation: current 1d volume > 2x 20-period average
        vol_ma = vol_ma_1d_aligned[i]
        volume_confirm = volume > 2 * vol_ma
        
        # Weekly RSI filter: >50 for bullish bias, <50 for bearish bias
        rsi_bullish = rsi_1w_aligned[i] > 50
        rsi_bearish = rsi_1w_aligned[i] < 50
        
        if position == 0:
            if volume_confirm:
                # Long: price breaks above Donchian high with bullish weekly bias
                if price > donchian_high[i] and rsi_bullish:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below Donchian low with bearish weekly bias
                elif price < donchian_low[i] and rsi_bearish:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit when price crosses Donchian midpoint
            exit_signal = False
            
            if position == 1:  # long position
                if price < donchian_mid[i]:
                    exit_signal = True
            
            elif position == -1:  # short position
                if price > donchian_mid[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian15_Breakout_1wRSI_Filter_1dVolumeSpike"
timeframe = "6h"
leverage = 1.0
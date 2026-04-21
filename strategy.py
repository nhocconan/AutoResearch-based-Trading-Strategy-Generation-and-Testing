#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) Breakout + 1d RSI Filter + Volume Confirmation
# Long when price breaks above Donchian(20) high, 1d RSI < 40 (avoid overbought), and volume > 1.5x average
# Short when price breaks below Donchian(20) low, 1d RSI > 60 (avoid oversold), and volume > 1.5x average
# Exit when price crosses Donchian midpoint
# RSI filter prevents buying strength/weakness, focusing on mean-reversion within trend
# Volume confirms breakout authenticity
# Target: 15-25 trades/year via strict RSI + volume + breakout confluence

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d RSI(14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    rsi_period = 14
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
    avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
    
    for i in range(rsi_period+1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[0] = 50  # First value neutral
    
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate Donchian(20) on 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(rsi_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price
        price = prices['close'].iloc[i]
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        vol_ma = vol_ma_1d_aligned[i]
        # Get current 1d volume (same for all 6h bars within the day)
        idx_1d = i // 4  # 4 six-hour bars per day
        if idx_1d >= len(df_1d):
            idx_1d = len(df_1d) - 1
        current_vol = df_1d['volume'].iloc[idx_1d]
        volume_confirm = current_vol > 1.5 * vol_ma
        
        # RSI filter: avoid extremes
        rsi_val = rsi_aligned[i]
        rsi_long_filter = rsi_val < 40  # Not overbought
        rsi_short_filter = rsi_val > 60  # Not oversold
        
        if position == 0:
            if volume_confirm:
                # Long: price breaks above Donchian high with RSI < 40
                if price > donchian_high[i] and rsi_long_filter:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below Donchian low with RSI > 60
                elif price < donchian_low[i] and rsi_short_filter:
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

name = "6h_Donchian20_Breakout_1dRSI_Filter_Volume"
timeframe = "6h"
leverage = 1.0
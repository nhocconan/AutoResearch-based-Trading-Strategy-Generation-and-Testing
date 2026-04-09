#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian channel breakout for trend and 1w RSI for momentum confirmation
# - Uses 1w HTF for Donchian(20): price above/below 20-period high/low determines trend
# - Uses 1w HTF for RSI(14): RSI > 50 for bullish momentum, RSI < 50 for bearish momentum
# - In bullish trend (price > 20-period high): look for long entries when 1d close > 1d open (bullish daily candle)
# - In bearish trend (price < 20-period low): look for short entries when 1d close < 1d open (bearish daily candle)
# - Volume confirmation: current 1d volume > 1.5x 20-period average to avoid low-volume false signals
# - Fixed position size 0.25 to control drawdown
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)

name = "1d_1w_donchian_rsi_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Donchian channel (20 periods)
    period20_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w RSI (14 periods)
    delta = pd.Series(close_1w).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Align all HTF data to 1d timeframe (wait for completed HTF bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, period20_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, period20_low)
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi_values)
    
    # Pre-compute volume confirmation (20-period average for 1d)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(rsi_aligned[i]) or np.isnan(vol_ma_20[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend determination: price above/below Donchian channel
        bullish_trend = close[i] > donchian_high_aligned[i]
        bearish_trend = close[i] < donchian_low_aligned[i]
        
        # RSI momentum: >50 = bullish, <50 = bearish
        rsi_bullish = rsi_aligned[i] > 50
        rsi_bearish = rsi_aligned[i] < 50
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit conditions
            if bullish_trend and rsi_bullish:
                # In bullish trend with bullish momentum: hold
                signals[i] = position_size
            else:
                # Exit when trend or momentum changes
                position = 0
                signals[i] = 0.0
                
        elif position == -1:  # Short position
            # Exit conditions
            if bearish_trend and rsi_bearish:
                # In bearish trend with bearish momentum: hold
                signals[i] = -position_size
            else:
                # Exit when trend or momentum changes
                position = 0
                signals[i] = 0.0
        else:  # Flat
            # Entry logic based on trend, momentum, and volume
            if volume_confirmed:
                if bullish_trend and rsi_bullish and close[i] > open_price[i]:
                    # In bullish trend with bullish momentum and bullish daily candle: long
                    position = 1
                    signals[i] = position_size
                elif bearish_trend and rsi_bearish and close[i] < open_price[i]:
                    # In bearish trend with bearish momentum and bearish daily candle: short
                    position = -1
                    signals[i] = -position_size
    
    return signals
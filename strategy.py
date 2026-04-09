#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian breakout for trend and 1d RSI for mean reversion
# - Uses 1d HTF for Donchian channel (20-period high/low) to determine trend direction
# - Uses 1d HTF for RSI(14) to identify overbought/oversold conditions
# - In bullish trend (price above Donchian upper band): look for long entries when RSI < 30 (oversold pullback)
# - In bearish trend (price below Donchian lower band): look for short entries when RSI > 70 (overbought bounce)
# - Volume confirmation: current 12h volume > 1.5x 20-period average to avoid low-volume false signals
# - Fixed position size 0.25 to control drawdown
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1d_donchian_rsi_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channel (20 periods)
    period20_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_upper = period20_high
    donchian_lower = period20_low
    
    # Calculate 1d RSI (14 periods)
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Align all HTF data to 12h timeframe (wait for completed HTF bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # Pre-compute volume confirmation (20-period average for 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(rsi_aligned[i]) or np.isnan(vol_ma_20[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend determination: price relative to Donchian channel
        bullish_trend = close[i] > donchian_upper_aligned[i]
        bearish_trend = close[i] < donchian_lower_aligned[i]
        
        # RSI extremes: <30 = oversold, >70 = overbought
        oversold = rsi_aligned[i] < 30
        overbought = rsi_aligned[i] > 70
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit conditions
            if bullish_trend:
                # In bullish trend: exit when overbought or trend changes to bearish
                if overbought or bearish_trend:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            else:
                # Not in bullish trend: exit
                position = 0
                signals[i] = 0.0
                
        elif position == -1:  # Short position
            # Exit conditions
            if bearish_trend:
                # In bearish trend: exit when oversold or trend changes to bullish
                if oversold or bullish_trend:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
            else:
                # Not in bearish trend: exit
                position = 0
                signals[i] = 0.0
        else:  # Flat
            # Entry logic based on trend and RSI extremes
            if volume_confirmed:
                if bullish_trend and oversold:
                    # In bullish trend, daily oversold: long mean reversion
                    position = 1
                    signals[i] = position_size
                elif bearish_trend and overbought:
                    # In bearish trend, daily overbought: short mean reversion
                    position = -1
                    signals[i] = -position_size
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly volume confirmation and RSI filter.
# Uses weekly Donchian channels and RSI to avoid whipsaws in both bull and bear markets.
# Timeframe 1d reduces trade frequency to minimize fee drag while capturing medium-term trends.
# Volume confirmation ensures breakouts have conviction. RSI filter avoids overextended moves.
# Target: 30-100 total trades over 4 years (7-25/year) to stay within profitable range.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for multi-timeframe analysis
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 20-week Donchian channel on weekly
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly RSI for overbought/oversold conditions
    delta = pd.Series(df_1w['close']).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate weekly volume and its 20-period average
    volume_1w = df_1w['volume'].values
    volume_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align all data to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi_values)
    volume_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_20_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(rsi_aligned[i]) or np.isnan(volume_ma_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current daily volume > 1.5x weekly volume MA (adjusted for daily)
        # ~7 daily periods per week, so weekly MA/7 = approximate daily period MA
        volume_daily_approx_ma = volume_ma_20_1w_aligned[i] / 7
        volume_condition = volume[i] > (volume_daily_approx_ma * 1.5)
        
        # RSI conditions: avoid extreme overbought/oversold
        rsi_not_overbought = rsi_aligned[i] < 70
        rsi_not_oversold = rsi_aligned[i] > 30
        
        # Entry conditions: Donchian breakout with volume and RSI filter
        # Long when price breaks above Donchian high with volume and not overbought
        # Short when price breaks below Donchian low with volume and not oversold
        breakout_long = close[i] > donchian_high_aligned[i]
        breakout_short = close[i] < donchian_low_aligned[i]
        
        if position == 0:
            if breakout_long and volume_condition and rsi_not_overbought:
                position = 1
                signals[i] = position_size
            elif breakout_short and volume_condition and rsi_not_oversold:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price breaks below Donchian low or RSI becomes overbought
            if close[i] < donchian_low_aligned[i] or rsi_aligned[i] >= 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price breaks above Donchian high or RSI becomes oversold
            if close[i] > donchian_high_aligned[i] or rsi_aligned[i] <= 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Donchian_Breakout_Volume_RSI_Filter_v1"
timeframe = "1d"
leverage = 1.0
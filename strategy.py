#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R4/S4 breakout with 1d EMA50 trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R4 level in bull trend (close > 1d EMA50) with volume > 2.0x 20-period MA.
# Short when price breaks below Camarilla S4 level in bear trend (close < 1d EMA50) with volume spike.
# Uses discrete position sizing (0.25) to balance return and drawdown. Camarilla levels from daily pivot provide
# institutional support/resistance, ideal for 12h timeframe. Volume confirmation ensures institutional participation.
# 1d trend filter reduces whipsaw vs shorter MAs, improving performance in both bull and bear markets.
# Target: 75-150 total trades over 4 years (19-38/year).

name = "12h_Camarilla_R4S4_1dEMA50_Volume"
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
    
    # Get 1d data for pivot points and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d pivot points (using previous day's OHLC)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r4 = prev_high + 3 * (pivot - prev_low)
    s4 = prev_low - 3 * (prev_high - pivot)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align Camarilla levels and EMA to 12h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume regime: current 12h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Camarilla breakout conditions
        breakout_up = close_val > r4_aligned[i-1]  # break above previous period's R4
        breakout_down = close_val < s4_aligned[i-1]  # break below previous period's S4
        
        # Entry logic
        if position == 0:
            if is_bull_trend and breakout_up and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_bear_trend and breakout_down and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Camarilla breakout down OR trend reversal
            if breakout_down or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Camarilla breakout up OR trend reversal
            if breakout_up or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
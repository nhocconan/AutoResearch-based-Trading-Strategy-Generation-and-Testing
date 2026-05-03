#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above 4h Donchian upper band in bull trend (close > 12h EMA50) with volume > 2.0x 20-period MA.
# Short when price breaks below 4h Donchian lower band in bear trend (close < 12h EMA50) with volume spike.
# Uses discrete position sizing (0.30) to minimize fee churn. 12h EMA50 provides strong trend filter.
# Volume confirmation ensures institutional participation. Target: 75-200 total trades over 4 years.

name = "4h_Donchian20_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume regime: current 4h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_12h_aligned[i]
        upper_band = donchian_upper[i]
        lower_band = donchian_lower[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Donchian breakout conditions
        breakout_upper = close_val > upper_band
        breakout_lower = close_val < lower_band
        
        # Entry logic
        if position == 0:
            if is_bull_trend and breakout_upper and vol_spike:
                signals[i] = 0.30
                position = 1
            elif is_bear_trend and breakout_lower and vol_spike:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: price breaks below lower band OR trend reversal
            if close_val < lower_band or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: price breaks above upper band OR trend reversal
            if close_val > upper_band or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
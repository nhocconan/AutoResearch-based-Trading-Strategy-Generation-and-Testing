#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout (20-period) with 1-week EMA200 trend filter and volume confirmation.
# Donchian breakouts capture momentum moves; EMA200 ensures alignment with longer-term trend.
# Volume confirms institutional participation. Designed to work in both bull and bear markets
# by filtering trades with the weekly trend. Target: 15-25 trades per year to minimize fee drag.

name = "1d_Donchian20_1wEMA200_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1w data ONCE before loop for EMA200 trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === 1w EMA200 for trend direction ===
    close_1w = df_1w['close'].values
    ema_200 = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200)
    
    # === 1d Donchian channels (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):  # Start after Donchian warmup
        # Get values
        close_val = prices['close'].iloc[i]
        ema_val = ema_200_aligned[i]
        upper = donch_high[i]
        lower = donch_low[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_val) or np.isnan(upper) or np.isnan(lower) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price closes above upper Donchian band with uptrend and volume
            if close_val > upper and close_val > ema_val and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short breakdown: price closes below lower Donchian band with downtrend and volume
            elif close_val < lower and close_val < ema_val and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        
        elif position == 1:
            # Long exit: price closes below lower Donchian band or trend reversal
            if close_val < lower or close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above upper Donchian band or trend reversal
            if close_val > upper or close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
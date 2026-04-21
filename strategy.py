#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for weekly trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA10 for long-term trend
    close_1w = df_1w['close'].values
    ema_10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # Weekly EMA30 for medium-term trend
    ema_30_1w = pd.Series(close_1w).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema_30_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_30_1w)
    
    # Daily Donchian(20) for breakout signals
    high = prices['high'].values
    low = prices['low'].values
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(prices['close'].values, 1))
    tr3 = np.abs(low - np.roll(prices['close'].values, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume / 50-period average volume
    vol_ma_50 = pd.Series(prices['volume'].values).rolling(window=50, min_periods=50).mean().values
    vol_ratio = prices['volume'].values / vol_ma_50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_10_1w_aligned[i]) or np.isnan(ema_30_1w_aligned[i]) or 
            np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_10_1w_aligned[i]
        ema_medium = ema_30_1w_aligned[i]
        upper_band = donch_high[i]
        lower_band = donch_low[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr_14[i]
        
        # Trend filter: weekly EMA10 > EMA30 for uptrend, < for downtrend
        is_uptrend = ema_trend > ema_medium
        is_downtrend = ema_trend < ema_medium
        
        if position == 0:
            # Enter long: break above Donchian high + uptrend + volume surge + moderate volatility
            if (price_close > upper_band and 
                is_uptrend and 
                vol_ratio_val > 2.0 and 
                atr_val > 0.5 * np.nanmedian(atr_14[max(0, i-50):i+1]) and
                atr_val < 3.0 * np.nanmedian(atr_14[max(0, i-50):i+1])):
                signals[i] = 0.25
                position = 1
            # Enter short: break below Donchian low + downtrend + volume surge + moderate volatility
            elif (price_close < lower_band and 
                  is_downtrend and 
                  vol_ratio_val > 2.0 and 
                  atr_val > 0.5 * np.nanmedian(atr_14[max(0, i-50):i+1]) and
                  atr_val < 3.0 * np.nanmedian(atr_14[max(0, i-50):i+1])):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: reverse breakout or volatility extremes
            if position == 1 and (price_close < lower_band or 
                                  vol_ratio_val < 0.5 or 
                                  atr_val > 4.0 * np.nanmedian(atr_14[max(0, i-50):i+1])):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close > upper_band or 
                                     vol_ratio_val < 0.5 or 
                                     atr_val > 4.0 * np.nanmedian(atr_14[max(0, i-50):i+1])):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WeeklyEMA_Crossover_DonchianBreakout_VolumeATR"
timeframe = "1d"
leverage = 1.0
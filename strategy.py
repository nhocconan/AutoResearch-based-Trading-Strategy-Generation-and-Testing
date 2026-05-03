#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation.
# Long when price breaks above Donchian upper band in bull trend (close > 1d EMA50) with volume > 2.0x 20-period MA.
# Short when price breaks below Donchian lower band in bear trend (close < 1d EMA50) with volume spike.
# ATR-based stoploss: exit long when price < highest high - 2.5*ATR, exit short when price < lowest low + 2.5*ATR.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 75-200 total trades over 4 years.

name = "4h_Donchian20_1dEMA50_Volume_ATR"
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume regime: current 4h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_high = 0.0  # for long position trailing stop
    lowest_low = 0.0    # for short position trailing stop
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_1d_aligned[i]
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Donchian breakout conditions
        breakout_upper = close_val > upper_band
        breakout_lower = close_val < lower_band
        
        # Update trailing stops
        if position == 1:  # long position
            highest_high = max(highest_high, high[i])
            # ATR stoploss: exit if price drops below highest_high - 2.5*ATR
            if close_val < highest_high - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                highest_high = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            lowest_low = min(lowest_low, low[i])
            # ATR stoploss: exit if price rises above lowest_low + 2.5*ATR
            if close_val > lowest_low + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                lowest_low = 0.0
            else:
                signals[i] = -0.25
        else:  # position == 0, look for entry
            if is_bull_trend and breakout_upper and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_high = high[i]
            elif is_bear_trend and breakout_lower and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_low = low[i]
    
    return signals
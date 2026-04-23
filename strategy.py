#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR-based volatility filter.
Long when price breaks above Donchian upper band and close > 1d EMA50 (uptrend) with ATR(14) > 0.5 * ATR(50) (sufficient volatility).
Short when price breaks below Donchian lower band and close < 1d EMA50 (downtrend) with ATR(14) > 0.5 * ATR(50).
Exit on opposite Donchian break or trend reversal. Uses 4h timeframe targeting 75-200 total trades over 4 years.
Donchian channels provide clear structure, 1d EMA50 filters medium-term trend, ATR volatility filter avoids choppy markets.
Designed to capture strong momentum moves while avoiding whipsaws in both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Donchian channels (20-period) on primary timeframe
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # ATR volatility filter (14 and 50 periods)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(atr14[i]) or np.isnan(atr50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_1d_aligned[i]
        upper_band = highest_high[i]
        lower_band = lowest_low[i]
        atr14_val = atr14[i]
        atr50_val = atr50[i]
        price = close[i]
        
        # Volatility filter: require sufficient volatility (ATR14 > 0.5 * ATR50)
        vol_filter = atr14_val > 0.5 * atr50_val
        
        if position == 0:
            # Long: price breaks above upper band AND price > 1d EMA50 (uptrend) AND volatility filter
            if (price > upper_band and price > ema50_val and vol_filter):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower band AND price < 1d EMA50 (downtrend) AND volatility filter
            elif (price < lower_band and price < ema50_val and vol_filter):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below lower band OR trend reversal
                if (price < lower_band or price < ema50_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above upper band OR trend reversal
                if (price > upper_band or price > ema50_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_1dEMA50_ATR_VolFilter"
timeframe = "4h"
leverage = 1.0
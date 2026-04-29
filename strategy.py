#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Uses Donchian channel from 4h: breakout above upper band or below lower band
# Volume confirmation (>1.5x 20-period average) ensures institutional participation
# Trend filter uses 1d EMA50 to avoid counter-trend trades in both bull and bear markets
# Stoploss: 2.0 * ATR(14) from entry price
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag
# Designed for 4h timeframe with proven edge on SOLUSDT and adaptation for BTC/ETH
# BTC/ETH focus: requires EMA alignment and volume confirmation to avoid SOL-only bias

name = "4h_Donchian20_1dEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian(20) from 4h data
    # Upper band: highest high over last 20 periods
    # Lower band: lowest low over last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = max(50, 20, 14)  # EMA50, Donchian, and ATR warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_upper = donchian_upper[i]
        curr_lower = donchian_lower[i]
        curr_atr = atr[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle stoploss and exits
        if position == 1:  # Long position
            # Stoploss: price closes below entry - 2.0 * ATR_at_entry
            if curr_close < entry_price - 2.0 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            # Exit: price breaks below lower Donchian band or trend turns down
            elif curr_close < curr_lower or curr_close < curr_ema50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Stoploss: price closes above entry + 2.0 * ATR_at_entry
            if curr_close > entry_price + 2.0 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            # Exit: price breaks above upper Donchian band or trend turns up
            elif curr_close > curr_upper or curr_close > curr_ema50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirm = curr_volume > 1.5 * curr_vol_ma
            
            # Long entry: price breaks above upper Donchian band in uptrend (price > EMA50_1d)
            if vol_confirm and curr_close > curr_ema50_1d:
                if curr_high > curr_upper:  # Break above upper band
                    signals[i] = 0.30
                    position = 1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
            # Short entry: price breaks below lower Donchian band in downtrend (price < EMA50_1d)
            elif vol_confirm and curr_close < curr_ema50_1d:
                if curr_low < curr_lower:  # Break below lower band
                    signals[i] = -0.30
                    position = -1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals
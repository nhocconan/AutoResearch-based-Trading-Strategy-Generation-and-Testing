#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Uses Donchian channel from 20-period high/low on 1d timeframe
# Breakout above 20d high in uptrend (price > 1w EMA50) or below 20d low in downtrend (price < 1w EMA50)
# Volume confirmation (>1.5x 20-period average) ensures institutional participation
# Trend filter uses 1w EMA50 to avoid counter-trend trades in both bull and bear markets
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag
# Designed for 1d timeframe to capture major swings with low frequency
# BTC/ETH focus: requires EMA alignment and volume confirmation to avoid SOL-only bias

name = "1d_Donchian20_1wEMA50_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian(20) channels on 1d
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels (already on 1d, no alignment needed)
    # But we need to shift by 1 to avoid look-ahead (use previous bar's channel)
    donchian_high = np.roll(donchian_high, 1)
    donchian_low = np.roll(donchian_low, 1)
    donchian_high[0] = np.nan
    donchian_low[0] = np.nan
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    # Calculate ATR(14) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    start_idx = max(50, 20, 14)  # EMA50, Donchian, ATR warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema50_1w = ema_50_1w_aligned[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_atr = atr[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle stoploss and exits
        if position == 1:  # Long position
            # Stoploss: price closes below entry - 2.5 * ATR_at_entry
            if curr_close < entry_price - 2.5 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            # Exit: price breaks below Donchian low or trend turns down
            elif curr_close < curr_donchian_low or curr_close < curr_ema50_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: price closes above entry + 2.5 * ATR_at_entry
            if curr_close > entry_price + 2.5 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            # Exit: price breaks above Donchian high or trend turns up
            elif curr_close > curr_donchian_high or curr_close > curr_ema50_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirm = curr_volume > 1.5 * curr_vol_ma
            
            # Long entry: price breaks above Donchian high in uptrend (price > EMA50_1w)
            if vol_confirm and curr_close > curr_ema50_1w:
                if curr_high > curr_donchian_high:  # Break above Donchian high
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
            # Short entry: price breaks below Donchian low in downtrend (price < EMA50_1w)
            elif vol_confirm and curr_close < curr_ema50_1w:
                if curr_low < curr_donchian_low:  # Break below Donchian low
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals
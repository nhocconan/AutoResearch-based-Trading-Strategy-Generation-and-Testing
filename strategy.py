#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike
# Uses Donchian channel from 1d: breakout above upper or below lower with continuation
# Volume confirmation (>2.0x 24-period average) ensures institutional participation
# Trend filter uses 1w EMA50 to avoid counter-trend trades in both bull and bear markets
# Designed for 1d timeframe to capture major swings with controlled trade frequency (~10-25 trades/year)
# BTC/ETH focus: requires EMA alignment and volume confirmation, avoids SOL-only bias

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
    
    # Calculate ATR(14) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 24-period average volume for confirmation
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = max(50, 24, 14)  # EMA50, volume MA, and ATR warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema50_1w = ema_50_1w_aligned[i]
        curr_atr = atr[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_24[i]
        
        # Calculate Donchian channels from prior 20 periods
        if i >= 20:
            lookback_high = np.max(high[i-20:i])
            lookback_low = np.min(low[i-20:i])
        else:
            signals[i] = 0.0
            continue
        
        # Handle stoploss and exits
        if position == 1:  # Long position
            # Stoploss: price closes below entry - 2.5 * ATR_at_entry
            if curr_close < entry_price - 2.5 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            # Exit: price breaks below lower Donchian or trend turns down
            elif curr_close < lookback_low or curr_close < curr_ema50_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Stoploss: price closes above entry + 2.5 * ATR_at_entry
            if curr_close > entry_price + 2.5 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            # Exit: price breaks above upper Donchian or trend turns up
            elif curr_close > lookback_high or curr_close > curr_ema50_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 24-period average
            vol_confirm = curr_volume > 2.0 * curr_vol_ma
            
            # Long entry: price breaks above upper Donchian in uptrend (price > EMA50_1w)
            if vol_confirm and curr_close > curr_ema50_1w:
                if curr_high > lookback_high:  # Break above upper Donchian
                    signals[i] = 0.30
                    position = 1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
            # Short entry: price breaks below lower Donchian in downtrend (price < EMA50_1w)
            elif vol_confirm and curr_close < curr_ema50_1w:
                if curr_low < lookback_low:  # Break below lower Donchian
                    signals[i] = -0.30
                    position = -1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals
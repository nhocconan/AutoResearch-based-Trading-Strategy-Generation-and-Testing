#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Uses Donchian channel from prior 4h period: long on break above upper band in uptrend, short on break below lower band in downtrend
# Volume confirmation (>1.5x 20-period average) ensures institutional participation
# Trend filter uses 12h EMA50 to avoid counter-trend trades in both bull and bear markets
# ATR-based stoploss (2.5x ATR) manages risk and reduces whipsaw
# Designed for 4h timeframe to capture medium-term swings with controlled trade frequency (~20-40 trades/year)

name = "4h_Donchian20_12hEMA50_VolumeConfirm_ATRStop_v1"
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
    
    # Get 12h data for EMA50 trend filter (HTF = 12h)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 4h data for Donchian channel calculation (HTF = 4h)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channel (20-period) from prior 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (delayed by one 4h bar for look-ahead avoidance)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    
    # Calculate ATR(14) for stoploss and volatility filter
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period average volume for confirmation (on 4h data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = max(20, 14)  # Donchian and ATR warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema50_12h = ema_50_12h_aligned[i]
        curr_donchian_upper = donchian_upper_aligned[i]
        curr_donchian_lower = donchian_lower_aligned[i]
        curr_atr = atr[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle stoploss and exits
        if position == 1:  # Long position
            # Stoploss: price closes below entry - 2.5 * ATR_at_entry
            if curr_close < entry_price - 2.5 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            # Exit: price breaks below Donchian lower or trend turns down
            elif curr_close < curr_donchian_lower or curr_close < curr_ema50_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Stoploss: price closes above entry + 2.5 * ATR_at_entry
            if curr_close > entry_price + 2.5 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            # Exit: price breaks above Donchian upper or trend turns up
            elif curr_close > curr_donchian_upper or curr_close > curr_ema50_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirm = curr_volume > 1.5 * curr_vol_ma
            
            # Long entry: price breaks above Donchian upper in uptrend (price > EMA50)
            if vol_confirm and curr_close > curr_ema50_12h:
                if curr_high > curr_donchian_upper:  # Break above upper band
                    signals[i] = 0.30
                    position = 1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
            # Short entry: price breaks below Donchian lower in downtrend (price < EMA50)
            elif vol_confirm and curr_close < curr_ema50_12h:
                if curr_low < curr_donchian_lower:  # Break below lower band
                    signals[i] = -0.30
                    position = -1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals
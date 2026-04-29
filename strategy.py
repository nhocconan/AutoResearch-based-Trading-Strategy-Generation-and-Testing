#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Uses Donchian channel from prior 1d period: long on break above upper band in uptrend, short on break below lower band in downtrend
# Volume confirmation (>1.5x 20-period average) ensures institutional participation
# Designed for 1d timeframe to capture long-term swings with controlled trade frequency (~10-25 trades/year)
# Works in both bull and bear markets by aligning with 1w trend (EMA50) to avoid counter-trend trades
# Includes ATR-based stoploss (2.5x ATR) to manage risk during adverse moves

name = "1d_Donchian20_1wEMA50_VolumeConfirm_ATRStop_v1"
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
    
    # Get 1d data for Donchian calculation (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian(20) from prior 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper band = highest high of prior 20 periods, Lower band = lowest low of prior 20 periods
    upper_band = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_band = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 1d timeframe (delayed by one 1d bar for look-ahead avoidance)
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # Calculate 20-period average volume for confirmation (on 1d data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) for stoploss
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = 0  # First value has no previous close
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # Track entry price for stoploss
    
    start_idx = max(20, 14)  # Volume MA and ATR warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(upper_band_aligned[i]) or 
            np.isnan(lower_band_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema50_1w = ema_50_1w_aligned[i]
        curr_upper = upper_band_aligned[i]
        curr_lower = lower_band_aligned[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        curr_atr = atr[i]
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Stoploss: price drops below entry_price - 2.5 * ATR
            if curr_low < entry_price - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below lower Donchian band or trend turns down
            elif curr_close < curr_lower or curr_close < curr_ema50_1w:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: price rises above entry_price + 2.5 * ATR
            if curr_high > entry_price + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above upper Donchian band or trend turns up
            elif curr_close > curr_upper or curr_close > curr_ema50_1w:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirm = curr_volume > 1.5 * curr_vol_ma
            
            # Long entry: price breaks above upper Donchian band in uptrend
            if vol_confirm and curr_close > curr_ema50_1w:
                if curr_high > curr_upper:  # Break above upper band
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
            # Short entry: price breaks below lower Donchian band in downtrend
            elif vol_confirm and curr_close < curr_ema50_1w:
                if curr_low < curr_lower:  # Break below lower band
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals
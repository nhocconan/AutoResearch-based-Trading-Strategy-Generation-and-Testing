#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Camarilla levels provide precise support/resistance; R3/S3 are strong breakout levels
# Trend filter ensures we only trade breakouts in direction of 1d EMA34
# Volume confirmation (>2.0x 20-period average) reduces false breakouts
# Designed for ~20-50 trades/year on 4h timeframe to minimize fee drag
# Works in both bull and bear via 1d EMA34 trend filter - only trades in direction of higher timeframe momentum

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get 1d data for EMA34 trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels for 4h timeframe
    # Camarilla uses previous day's OHLC
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Get previous day's OHLC from 4h data (we need daily OHLC, so we'll resample conceptually but using actual data)
    # Since we're on 4h timeframe, we need to get the daily OHLC from the 1d dataframe we already have
    prev_close_1d = df_1d['close'].shift(1).values  # Previous day's close
    prev_high_1d = df_1d['high'].shift(1).values    # Previous day's high
    prev_low_1d = df_1d['low'].shift(1).values      # Previous day's low
    prev_open_1d = df_1d['open'].shift(1).values    # Previous day's open
    
    # Typical price for Camarilla calculation
    typical_price = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    range_hl = prev_high_1d - prev_low_1d
    
    # Calculate Camarilla levels
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    camarilla_r3 = prev_close_1d + range_hl * 1.1 / 4
    camarilla_s3 = prev_close_1d - range_hl * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate ATR (14-period) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = 34  # EMA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_open = open_price[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_atr = atr[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: stoploss hit or price closes below S3 (failed breakout)
            if curr_close < entry_price - 1.5 * curr_atr or curr_close < curr_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: stoploss hit or price closes above R3 (failed breakout)
            if curr_close > entry_price + 1.5 * curr_atr or curr_close > curr_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 20-period average
            vol_confirm = curr_volume > 2.0 * curr_vol_ma
            
            # Long entry when price > 1d EMA34 (bullish regime) AND price breaks above R3 with volume confirmation
            if curr_close > curr_ema34_1d and curr_close > curr_r3 and vol_confirm:
                # Additional confirmation: close near high of bar (strong breakout)
                if curr_close > (curr_high + curr_low) / 2:  # Close in upper half of bar
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
            # Short entry when price < 1d EMA34 (bearish regime) AND price breaks below S3 with volume confirmation
            elif curr_close < curr_ema34_1d and curr_close < curr_s3 and vol_confirm:
                # Additional confirmation: close near low of bar (strong breakout)
                if curr_close < (curr_high + curr_low) / 2:  # Close in lower half of bar
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals
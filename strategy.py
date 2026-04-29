#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Camarilla pivot levels provide high-probability reversal/breakout levels derived from prior day's range
# In bull markets (price > 1d EMA34), we look for breaks above R3 with volume confirmation for longs
# In bear markets (price < 1d EMA34), we look for breaks below S3 with volume confirmation for shorts
# Uses volume > 2.0x 24-period average for confirmation to reduce false signals
# Designed for ~12-37 trades/year on 12h timeframe to minimize fee drag while capturing momentum
# Works in both bull and bear via 1d EMA34 trend filter - only trades in direction of higher timeframe momentum

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
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
    
    # Get 1d data for EMA34 trend filter and Camarilla pivots (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R4, R3, R2, R1, PP, S1, S2, S3, S4
    # R3 = close + 1.1*(high-low)/2
    # S3 = close - 1.1*(high-low)/2
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate ATR (14-period) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 24-period average volume for confirmation
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = 24  # volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_24[i])):
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
        curr_vol_ma = vol_ma_24[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: stoploss hit or price drops below R3 (failed breakout)
            if curr_close < entry_price - 1.5 * curr_atr or curr_close < curr_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: stoploss hit or price rises above S3 (failed breakout)
            if curr_close > entry_price + 1.5 * curr_atr or curr_close > curr_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 24-period average
            vol_confirm = curr_volume > 2.0 * curr_vol_ma
            
            # Long entry when price > 1d EMA34 (bullish regime) AND breaks above R3 with volume confirmation
            if curr_close > curr_ema34_1d and curr_close > curr_r3 and vol_confirm:
                # Additional confirmation: close near high of bar (strong bullish close)
                if curr_close > (curr_high + curr_low) / 2:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
            # Short entry when price < 1d EMA34 (bearish regime) AND breaks below S3 with volume confirmation
            elif curr_close < curr_ema34_1d and curr_close < curr_s3 and vol_confirm:
                # Additional confirmation: close near low of bar (strong bearish close)
                if curr_close < (curr_high + curr_low) / 2:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume spike
# Uses Camarilla pivot levels from 1d: breakout at R4/S4 with continuation, mean reversion at R3/S3
# Volume confirmation (>2.0x 20-period average) ensures institutional participation
# Trend filter uses 1d EMA50 to avoid counter-trend trades in both bull and bear markets
# Designed for 4h timeframe to capture medium-term swings with controlled trade frequency (~20-40 trades/year)

name = "4h_Camarilla_R3S3_Breakout_1dEMA50_VolumeSpike_v1"
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
    
    # Get 1d data for EMA50 trend filter and Camarilla pivot calculation (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla pivot levels from prior 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot formula
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Resistance levels
    r1 = pivot + (range_1d * 1.1 / 12)
    r2 = pivot + (range_1d * 1.1 / 6)
    r3 = pivot + (range_1d * 1.1 / 4)
    r4 = pivot + (range_1d * 1.1 / 2)
    
    # Support levels
    s1 = pivot - (range_1d * 1.1 / 12)
    s2 = pivot - (range_1d * 1.1 / 6)
    s3 = pivot - (range_1d * 1.1 / 4)
    s4 = pivot - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 1d timeframe (delayed by one 1d bar for look-ahead avoidance)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate ATR(14) for stoploss and volatility filter
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
    
    start_idx = max(50, 20, 14)  # EMA50, volume MA, and ATR warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_r3 = r3_aligned[i]
        curr_r4 = r4_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_s4 = s4_aligned[i]
        curr_atr = atr[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle stoploss and exits
        if position == 1:  # Long position
            # Stoploss: price closes below entry - 2.0 * ATR_at_entry
            if curr_close < entry_price - 2.0 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            # Exit: price breaks below S3 or trend turns down
            elif curr_close < curr_s3 or curr_close < curr_ema50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: price closes above entry + 2.0 * ATR_at_entry
            if curr_close > entry_price + 2.0 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            # Exit: price breaks above R3 or trend turns up
            elif curr_close > curr_r3 or curr_close > curr_ema50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 20-period average
            vol_confirm = curr_volume > 2.0 * curr_vol_ma
            
            # Long entry: price breaks above R4 in uptrend (price > EMA50)
            if vol_confirm and curr_close > curr_ema50_1d:
                if curr_high > curr_r4:  # Break above R4
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
            # Short entry: price breaks below S4 in downtrend (price < EMA50)
            elif vol_confirm and curr_close < curr_ema50_1d:
                if curr_low < curr_s4:  # Break below S4
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals
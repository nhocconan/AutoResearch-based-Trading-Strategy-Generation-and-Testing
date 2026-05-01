#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation.
# Long when price breaks above R3 AND 12h EMA50 uptrend AND volume > 2.0x 20-period median.
# Short when price breaks below S3 AND 12h EMA50 downtrend AND volume > 2.0x 20-period median.
# Uses ATR-based stoploss: exit long if price < highest_high_since_entry - 2.5*ATR(14),
# exit short if price > lowest_low_since_entry + 2.5*ATR(14).
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to avoid fee drag.
# Uses 12h HTF for trend filter to ensure alignment with major market cycles and reduce whipsaw.
# Camarilla levels calculated from prior 12h bar (H12, L12, C12) to avoid look-ahead.

name = "6h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike_ATR_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter (loaded once before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 14-period ATR for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Camarilla levels from prior 12h bar (H12, L12, C12)
    # H12 = high of prior 12h bar, L12 = low of prior 12h bar, C12 = close of prior 12h bar
    df_12h_for_camarilla = get_htf_data(prices, '12h')
    if len(df_12h_for_camarilla) < 2:
        return np.zeros(n)
    
    # Get prior 12h bar HLC (shifted by 1 to avoid look-ahead)
    H12 = df_12h_for_camarilla['high'].shift(1).values
    L12 = df_12h_for_camarilla['low'].shift(1).values
    C12 = df_12h_for_camarilla['close'].shift(1).values
    
    # Align to 6h timeframe (each 12h bar = 2x 6h bars)
    H12_aligned = align_htf_to_ltf(prices, df_12h_for_camarilla, H12)
    L12_aligned = align_htf_to_ltf(prices, df_12h_for_camarilla, L12)
    C12_aligned = align_htf_to_ltf(prices, df_12h_for_camarilla, C12)
    
    # Calculate Camarilla levels: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    R3 = C12_aligned + (H12_aligned - L12_aligned) * 1.1 / 4
    S3 = C12_aligned - (H12_aligned - L12_aligned) * 1.1 / 4
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup for EMA, ATR, and volume median
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(R3[i]) or 
            np.isnan(S3[i]) or 
            np.isnan(vol_median_20[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_atr = atr[i]
        
        # Trend filter: 12h EMA50 direction
        uptrend = curr_close > ema_50_12h_aligned[i]
        downtrend = curr_close < ema_50_12h_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 2.0)
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above R3 AND uptrend AND volume spike
            if curr_close > R3[i] and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            # Short: Price breaks below S3 AND downtrend AND volume spike
            elif curr_close < S3[i] and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high since entry
            if curr_close > highest_since_entry:
                highest_since_entry = curr_close
            
            # Exit conditions: ATR stoploss OR Camarilla S3 break OR trend reversal
            stop_price = highest_since_entry - 2.5 * curr_atr
            if curr_close < stop_price or curr_close < S3[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_close < lowest_since_entry:
                lowest_since_entry = curr_close
            
            # Exit conditions: ATR stoploss OR Camarilla R3 break OR trend reversal
            stop_price = lowest_since_entry + 2.5 * curr_atr
            if curr_close > stop_price or curr_close > R3[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
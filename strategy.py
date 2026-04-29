#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R4/S4 breakout with 1w Supertrend filter and volume spike
# Uses Camarilla pivot levels from 1d: breakout at R4/S4 (stronger than R3/S3) with continuation
# Volume confirmation (>1.5x 20-period average) reduces false breakouts
# Trend filter uses 1w Supertrend(ATR=10, mult=3.0) to capture major trend direction
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag
# Designed for 1d timeframe with tighter entry conditions to improve win rate and reduce overtrading

name = "1d_Camarilla_R4_S4_Breakout_1wSupertrend_Volume"
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
    
    # Get 1w data for Supertrend trend filter (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w Supertrend for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = pd.Series(high_1w - low_1w)
    tr2 = pd.Series(np.abs(high_1w - np.roll(close_1w, 1)))
    tr3 = pd.Series(np.abs(low_1w - np.roll(close_1w, 1)))
    tr_1w = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = pd.Series(tr_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2_1w = (high_1w + low_1w) / 2
    upper_band_1w = hl2_1w + (3.0 * atr_1w)
    lower_band_1w = hl2_1w - (3.0 * atr_1w)
    
    upper_band_1w = pd.Series(upper_band_1w)
    lower_band_1w = pd.Series(lower_band_1w)
    
    for i in range(1, len(upper_band_1w)):
        if close_1w[i-1] <= upper_band_1w[i-1]:
            upper_band_1w[i] = min(upper_band_1w[i], upper_band_1w[i-1])
        else:
            upper_band_1w[i] = upper_band_1w[i]
            
        if close_1w[i-1] >= lower_band_1w[i-1]:
            lower_band_1w[i] = max(lower_band_1w[i], lower_band_1w[i-1])
        else:
            lower_band_1w[i] = lower_band_1w[i]
    
    supertrend_1w = np.where(close_1w > upper_band_1w, -1,
                     np.where(close_1w < lower_band_1w, 1, 0))
    # Forward fill to maintain trend direction
    supertrend_1w = pd.Series(supertrend_1w).replace(0, np.nan).ffill().fillna(0).values
    supertrend_1w_aligned = align_htf_to_ltf(prices, df_1w, supertrend_1w)
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from prior 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot formula
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Resistance levels (focus on R4 for stronger breakout)
    r4 = pivot + (range_1d * 1.1 / 2)
    # Support levels (focus on S4 for stronger breakout)
    s4 = pivot - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 1d timeframe (delayed by one 1d bar for look-ahead avoidance)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
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
    
    start_idx = max(50, 20, 14)  # Supertrend, volume MA, and ATR warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_1w_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_supertrend_1w = supertrend_1w_aligned[i]
        curr_r4 = r4_aligned[i]
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
            # Exit: price breaks below S4 or trend turns down
            elif curr_close < curr_s4 or curr_supertrend_1w == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Stoploss: price closes above entry + 2.0 * ATR_at_entry
            if curr_close > entry_price + 2.0 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            # Exit: price breaks above R4 or trend turns up
            elif curr_close > curr_r4 or curr_supertrend_1w == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirm = curr_volume > 1.5 * curr_vol_ma
            
            # Long entry: price breaks above R4 in uptrend (Supertrend == 1)
            if vol_confirm and curr_supertrend_1w == 1:
                if curr_high > curr_r4:  # Break above R4
                    signals[i] = 0.30
                    position = 1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
            # Short entry: price breaks below S4 in downtrend (Supertrend == -1)
            elif vol_confirm and curr_supertrend_1w == -1:
                if curr_low < curr_s4:  # Break below S4
                    signals[i] = -0.30
                    position = -1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals
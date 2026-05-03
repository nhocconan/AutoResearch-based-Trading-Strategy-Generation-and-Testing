#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w trend filter and volume confirmation.
# Uses 1w EMA13, EMA8, EMA5 for trend direction (long when EMA5 > EMA8 > EMA13, short when reverse).
# Entry: price crosses above/below Alligator jaws (EMA13) with volume > 1.5x 20-period MA.
# Exit: ATR(14) trailing stop (2.5x ATR) or Alligator convergence (EMA8 crosses EMA13).
# Discrete sizing 0.25. Target: 30-100 total trades over 4 years (7-25/year).
# Alligator identifies trend exhaustion; 1w filter ensures alignment with higher timeframe;
# volume confirmation reduces false signals. Works in bull via trend continuation and
# in bear via counter-trend retraces with higher timeframe alignment.

name = "1d_WilliamsAlligator_1wTrend_Volume_ATR"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 13:
        return np.zeros(n)
    
    # Calculate 1w EMA5, EMA8, EMA13 for Alligator
    close_1w = df_1w['close'].values
    ema5_1w = pd.Series(close_1w).ewm(span=5, min_periods=5, adjust=False).mean().values
    ema8_1w = pd.Series(close_1w).ewm(span=8, min_periods=8, adjust=False).mean().values
    ema13_1w = pd.Series(close_1w).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Align to 1d timeframe
    ema5_1w_aligned = align_htf_to_ltf(prices, df_1w, ema5_1w)
    ema8_1w_aligned = align_htf_to_ltf(prices, df_1w, ema8_1w)
    ema13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema13_1w)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[tr[0]], tr])  # same length as prices
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d Alligator (EMA5, EMA8, EMA13)
    ema5_1d = pd.Series(close).ewm(span=5, min_periods=5, adjust=False).mean().values
    ema8_1d = pd.Series(close).ewm(span=8, min_periods=8, adjust=False).mean().values
    ema13_1d = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Volume regime: current 1d volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # for long ATR stop
    lowest_since_entry = 0.0   # for short ATR stop
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema5_1w_aligned[i]) or np.isnan(ema8_1w_aligned[i]) or np.isnan(ema13_1w_aligned[i]) or
            np.isnan(ema5_1d[i]) or np.isnan(ema8_1d[i]) or np.isnan(ema13_1d[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema5_1w_val = ema5_1w_aligned[i]
        ema8_1w_val = ema8_1w_aligned[i]
        ema13_1w_val = ema13_1w_aligned[i]
        ema5_1d_val = ema5_1d[i]
        ema8_1d_val = ema8_1d[i]
        ema13_1d_val = ema13_1d[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # Determine 1w trend regime
        is_uptrend_1w = (ema5_1w_val > ema8_1w_val) and (ema8_1w_val > ema13_1w_val)
        is_downtrend_1w = (ema5_1w_val < ema8_1w_val) and (ema8_1w_val < ema13_1w_val)
        
        # Determine 1d Alligator alignment
        is_alligator_long = (ema5_1d_val > ema8_1d_val) and (ema8_1d_val > ema13_1d_val)
        is_alligator_short = (ema5_1d_val < ema8_1d_val) and (ema8_1d_val < ema13_1d_val)
        is_alligator_converging = abs(ema8_1d_val - ema13_1d_val) < (0.001 * close_val)  # convergence filter
        
        # Update highest/lowest since entry for ATR stop
        if position == 1:
            highest_since_entry = max(highest_since_entry, high[i])
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, low[i]) if lowest_since_entry != 0 else low[i]
        
        # Entry logic
        if position == 0:
            # Long: price above Alligator lips (EMA5) with volume spike in 1w uptrend and 1d Alligator aligned
            if close_val > ema5_1d_val and vol_spike and is_uptrend_1w and is_alligator_long:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = high[i]
            # Short: price below Alligator lips (EMA5) with volume spike in 1w downtrend and 1d Alligator aligned
            elif close_val < ema5_1d_val and vol_spike and is_downtrend_1w and is_alligator_short:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = low[i]
        elif position == 1:
            # Long exit: ATR stoploss OR price below Alligator jaws (EMA13) OR 1w trend turns down OR Alligator converging
            atr_stop = highest_since_entry - (2.5 * atr_val)
            if close_val < atr_stop or close_val < ema13_1d_val or not is_uptrend_1w or is_alligator_converging:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: ATR stoploss OR price above Alligator jaws (EMA13) OR 1w trend turns up OR Alligator converging
            atr_stop = lowest_since_entry + (2.5 * atr_val)
            if close_val > atr_stop or close_val > ema13_1d_val or not is_downtrend_1w or is_alligator_converging:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals
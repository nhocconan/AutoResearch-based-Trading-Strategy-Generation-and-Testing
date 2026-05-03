#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1w EMA34 trend filter and volume confirmation.
# Uses 1w EMA34 for trend direction (long only when price > EMA34, short only when price < EMA34).
# Entry: Williams %R(14) crosses above -20 from below (long) or below -80 from above (short)
#        with volume > 1.5x 20-period MA and trend alignment.
# Exit: ATR(14) trailing stop (2.5x ATR) or Williams %R crosses opposite extreme (-80 for long, -20 for short).
# Discrete sizing 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# Williams %R identifies overbought/oversold conditions; 1w EMA34 filters counter-trend trades in higher timeframe;
# volume confirmation reduces false signals. Works in bull via pullback longs in uptrend
# and in bear via bounce shorts in downtrend.

name = "6h_WilliamsR_1wEMA34_Volume_ATR"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[tr[0]], tr])  # same length as prices
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume regime: current 6h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long ATR stop
    lowest_since_entry = 0.0   # for short ATR stop
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_34_1w_aligned[i]
        wr = williams_r[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # Determine trend regime
        is_uptrend = close_val > ema_trend
        is_downtrend = close_val < ema_trend
        
        # Update highest/lowest since entry for ATR stop
        if position == 1:
            highest_since_entry = max(highest_since_entry, high[i])
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, low[i]) if lowest_since_entry != 0 else low[i]
        
        # Williams %R extremes
        wr_overbought = wr > -20
        wr_oversold = wr < -80
        
        # Entry logic
        if position == 0:
            # Long: Williams %R crosses above -20 from below (exit oversold) with volume spike in uptrend
            if i > 0 and williams_r[i-1] <= -20 and wr > -20 and vol_spike and is_uptrend:
                signals[i] = 0.25
                position = 1
                highest_since_entry = high[i]
            # Short: Williams %R crosses below -80 from above (exit overbought) with volume spike in downtrend
            elif i > 0 and williams_r[i-1] >= -80 and wr < -80 and vol_spike and is_downtrend:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = low[i]
        elif position == 1:
            # Long exit: ATR stoploss OR Williams %R crosses below -80 (re-enter oversold)
            atr_stop = highest_since_entry - (2.5 * atr_val)
            if close_val < atr_stop or wr < -80:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: ATR stoploss OR Williams %R crosses above -20 (re-enter overbought)
            atr_stop = lowest_since_entry + (2.5 * atr_val)
            if close_val > atr_stop or wr > -20:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals
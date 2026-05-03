#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d EMA34 trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. Long when %R crosses above -80 from below
# with volume spike in uptrend (price > EMA34). Short when %R crosses below -20 from above
# with volume spike in downtrend (price < EMA34). Uses ATR(14) trailing stop (2.0x ATR).
# Discrete sizing 0.25. Target: 75-200 total trades over 4 years (19-50/year).
# Williams %R works in ranging markets (mean reversion) and EMA34 filter avoids counter-trend trades.
# Volume confirmation reduces false signals. Effective in both bull (buy dips) and bear (sell rallies).

name = "6h_WilliamsR_1dEMA34_Volume_ATR"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[tr[0]], tr])  # same length as prices
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Williams %R(14) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume regime: current 6h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # for long ATR stop
    lowest_since_entry = 0.0   # for short ATR stop
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_34_1d_aligned[i]
        wr = williams_r[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # Determine trend regime
        is_uptrend = close_val > ema_trend
        is_downtrend = close_val < ema_trend
        
        # Williams %R signals: long when crosses above -80, short when crosses below -20
        wr_long_signal = (wr > -80) and (i == 0 or williams_r[i-1] <= -80)
        wr_short_signal = (wr < -20) and (i == 0 or williams_r[i-1] >= -20)
        
        # Update highest/lowest since entry for ATR stop
        if position == 1:
            highest_since_entry = max(highest_since_entry, high[i])
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, low[i]) if lowest_since_entry != 0 else low[i]
        
        # Entry logic
        if position == 0:
            # Long: Williams %R crosses above -80 with volume spike in uptrend
            if wr_long_signal and vol_spike and is_uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = high[i]
            # Short: Williams %R crosses below -20 with volume spike in downtrend
            elif wr_short_signal and vol_spike and is_downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = low[i]
        elif position == 1:
            # Long exit: ATR stoploss OR Williams %R crosses below -50 (momentum loss) OR trend turns down
            atr_stop = highest_since_entry - (2.0 * atr_val)
            wr_exit = wr < -50  # exit long when momentum fades
            if close_val < atr_stop or wr_exit or not is_uptrend:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: ATR stoploss OR Williams %R crosses above -50 (momentum loss) OR trend turns up
            atr_stop = lowest_since_entry + (2.0 * atr_val)
            wr_exit = wr > -50  # exit short when momentum fades
            if close_val > atr_stop or wr_exit or not is_downtrend:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals
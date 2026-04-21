#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R mean-reversion with 1w trend filter and volume confirmation.
# In strong 1-week trends (price > 1w EMA34), Williams %R extremes (-80/-20) signal overextension
# likely to reverse. Volume > 2x average confirms mean-reversion strength. Works in bull/bear
# via trend filter (only trade with trend). Target: 30-100 total trades over 4 years.
# Position size: 0.25 to manage risk during drawdowns.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Load 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate Williams %R (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    
    # Align Williams %R to current timeframe (1d)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1-week EMA (34-period) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation using 1d volume
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price_close = prices['close'].iloc[i]
        vol_1d_current = align_htf_to_ltf(prices, df_1d, vol_1d)[i]
        
        if position == 0:
            # Enter long: Williams %R < -80 (oversold) + volume spike + price > 1w EMA (uptrend)
            if (williams_r_aligned[i] < -80 and
                vol_1d_current > 2.0 * vol_ma_20_1d_aligned[i] and
                price_close > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R > -20 (overbought) + volume spike + price < 1w EMA (downtrend)
            elif (williams_r_aligned[i] > -20 and
                  vol_1d_current > 2.0 * vol_ma_20_1d_aligned[i] and
                  price_close < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: Williams %R returns to neutral range (-50) or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R rises above -50 or trend turns down
                if (williams_r_aligned[i] > -50) or (price_close < ema_34_1w_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: Williams %R falls below -50 or trend turns up
                if (williams_r_aligned[i] < -50) or (price_close > ema_34_1w_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WilliamsR_MeanReversion_1wEMA34_Volume_Spike"
timeframe = "1d"
leverage = 1.0
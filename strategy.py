#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Long: Close breaks above R3 AND price > 1d EMA34 (uptrend) AND volume > 1.8x 24-period MA
# Short: Close breaks below S3 AND price < 1d EMA34 (downtrend) AND volume > 1.8x 24-period MA
# Exit: Opposite Camarilla breakout or EMA34 trend reversal.
# Discrete sizing 0.25. Target: 60-140 total trades over 4 years (15-35/year).
# Camarilla levels provide precise intraday support/resistance; 1d EMA34 filters higher timeframe trend;
# volume confirmation ensures breakout validity. Works in bull via longs with trend alignment
# and in bear via shorts with trend alignment. 12h timeframe minimizes fee drag.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels for 12h (using prior 12h bar's OHLC)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    #          S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    # We use the prior completed 12h bar to avoid look-ahead
    prior_high = np.roll(high, 1)
    prior_low = np.roll(low, 1)
    prior_close = np.roll(close, 1)
    prior_high[0] = np.nan  # First bar has no prior
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    
    rang = prior_high - prior_low
    r3 = prior_close + 1.1 * rang
    s3 = prior_close - 1.1 * rang
    
    # Volume regime: current 12h volume > 1.8x 24-period MA
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.8 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_uptrend = close_val > ema_trend
        is_downtrend = close_val < ema_trend
        
        # Entry logic
        if position == 0:
            # Long: Close breaks above R3 AND uptrend AND volume spike
            if close_val > r3[i] and is_uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S3 AND downtrend AND volume spike
            elif close_val < s3[i] and is_downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close breaks below S3 OR trend turns down
            if close_val < s3[i] or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close breaks above R3 OR trend turns up
            if close_val > r3[i] or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
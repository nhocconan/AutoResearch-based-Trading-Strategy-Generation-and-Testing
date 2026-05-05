#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R extreme with 1d EMA34 trend filter and volume confirmation
# Long when Williams %R < -80 (oversold) AND 1d close > 1d EMA34 AND volume > 1.8x 24-period average
# Short when Williams %R > -20 (overbought) AND 1d close < 1d EMA34 AND volume > 1.8x 24-period average
# Exit when Williams %R crosses above -50 (for long) or below -50 (for short)
# Uses 12h primary timeframe with 1d HTF for trend filter and Williams %R for mean reversion
# Williams %R identifies exhaustion points; 1d EMA34 ensures trend alignment; volume confirms participation
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# Discrete sizing (0.25) limits fee churn and manages drawdown in ranging/choppy markets

name = "12h_WilliamsR_Extreme_1dEMA34_Trend_Volume"
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
    
    # Get 1d data ONCE before loop for trend filter and Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R on 1d data: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Williams %R = -100 * (HH - C) / (HH - LL) where HH is highest high, LL is lowest low over period
    lookback = 14
    hh_1d = pd.Series(df_1d['high'].values).rolling(window=lookback, min_periods=lookback).max().values
    ll_1d = pd.Series(df_1d['low'].values).rolling(window=lookback, min_periods=lookback).min().values
    # Avoid division by zero
    rr_1d = hh_1d - ll_1d
    williams_r_1d = np.where(rr_1d != 0, -100 * (hh_1d - df_1d['close'].values) / rr_1d, -50.0)
    
    # Align Williams %R to 12h timeframe (wait for 1d bar to close)
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # Volume confirmation: volume > 1.8x 24-period average (24*12h = 12 days)
    if len(volume) >= 24:
        vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
        volume_filter = volume > (1.8 * vol_ma_24)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(williams_r_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold) AND 1d close > 1d EMA34 AND volume spike
            if (williams_r_1d_aligned[i] < -80.0 and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R > -20 (overbought) AND 1d close < 1d EMA34 AND volume spike
            elif (williams_r_1d_aligned[i] > -20.0 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -50 (recovering from oversold)
            if williams_r_1d_aligned[i] > -50.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -50 (declining from overbought)
            if williams_r_1d_aligned[i] < -50.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
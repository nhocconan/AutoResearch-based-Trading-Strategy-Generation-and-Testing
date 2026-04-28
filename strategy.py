#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d EMA34 trend filter and volume spike confirmation.
# Enter long when Williams %R < -80 (oversold) with volume spike and price above 1d EMA34.
# Enter short when Williams %R > -20 (overbought) with volume spike and price below 1d EMA34.
# Uses discrete position sizing (0.25) to limit drawdown. Target: 12-37 trades/year.
# Williams %R identifies exhaustion points, volume confirms reversal conviction, EMA34 filters intermediate trend.
# Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets.

name = "6h_WilliamsR14_1dEMA34_Trend_VolumeSpike_v1"
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
    
    # Get 1d data for EMA34 (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 6h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.full(n, np.nan)
    for i in range(n):
        if highest_high[i] != lowest_low[i]:  # Avoid division by zero
            williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            williams_r[i] = -50  # Neutral when range is zero
    
    # Calculate 6h volume spike: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1d EMA34
        above_ema = close[i] > ema_34_1d_aligned[i]
        below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Williams %R conditions with volume confirmation
        long_signal = williams_r[i] < -80 and volume_spike[i]  # Oversold
        short_signal = williams_r[i] > -20 and volume_spike[i]  # Overbought
        
        # Exit conditions: opposite Williams %R level or trend reversal
        long_exit = williams_r[i] > -20 or below_ema  # Exit long when overbotted or trend turns down
        short_exit = williams_r[i] < -80 or above_ema  # Exit short when oversold or trend turns up
        
        # Handle entries and exits
        if long_signal and above_ema and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_signal and below_ema and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals
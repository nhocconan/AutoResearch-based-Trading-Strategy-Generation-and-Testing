# Solution
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams %R with 4h EMA trend filter and session filter (08-20 UTC)
# Long when Williams %R < -80 (oversold) + close > 4h EMA50 (uptrend) + within active session
# Short when Williams %R > -20 (overbought) + close < 4h EMA50 (downtrend) + within active session
# Exit when Williams %R crosses above -50 (for long) or below -50 (for short) or trend reverses
# Williams %R identifies overbought/oversold conditions, effective in ranging markets
# Designed for low trade frequency (~15-30/year) to minimize fee drain.
# Works in bull/bear by combining mean reversion with trend filter and session timing.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 50-period EMA on 4h close for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Williams %R on 1h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    hl_range = highest_high - lowest_low
    willr = np.where(hl_range != 0, ((highest_high - close) / hl_range) * -100, -50.0)
    
    # Pre-calculate session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(willr[i]) or 
            np.isnan(ema_50_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        hour = hours[i]
        in_session = (8 <= hour <= 20)  # UTC 8-20
        willr_val = willr[i]
        ema_val = ema_50_4h_aligned[i]
        
        if position == 0 and in_session:
            # Long conditions: Williams %R < -80 (oversold) + uptrend
            if willr_val < -80.0 and price > ema_val:
                signals[i] = 0.20
                position = 1
            # Short conditions: Williams %R > -20 (overbought) + downtrend
            elif willr_val > -20.0 and price < ema_val:
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit conditions: Williams %R crosses above -50 (for long) or below -50 (for short) or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Williams %R rises above -50 or trend turns down
                if willr_val > -50.0 or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Williams %R falls below -50 or trend turns up
                if willr_val < -50.0 or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_WilliamsR_4hEMA50_Session"
timeframe = "1h"
leverage = 1.0
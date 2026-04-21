#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams %R with 1d trend filter and volume confirmation.
# Williams %R(14) identifies overbought/oversold conditions.
# In 1d uptrend (close > EMA50): buy oversold (Williams %R < -80), sell at neutral (Williams %R > -50).
# In 1d downtrend (close < EMA50): sell overbought (Williams %R > -20), buy at neutral (Williams %R < -50).
# Uses volume > 1.5x 20-period average for confirmation. Active only during 08-20 UTC session.
# Target: 15-35 trades/year by combining mean reversion with trend filter and volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Williams %R (14-period)
    highest_high = prices['high'].rolling(window=14, min_periods=14).max()
    lowest_low = prices['low'].rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - prices['close']) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).fillna(-50).values
    
    # 1d trend filter: EMA50 on daily close
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 20-period average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready or outside session
        if np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        wr = williams_r[i]
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # 1d trend: price above/below EMA50
        is_uptrend = price > ema_50_1d_aligned[i]
        is_downtrend = price < ema_50_1d_aligned[i]
        
        if position == 0:
            if volume_confirm:
                if is_uptrend and wr < -80:
                    # Oversold in uptrend: go long
                    signals[i] = 0.20
                    position = 1
                elif is_downtrend and wr > -20:
                    # Overbought in downtrend: go short
                    signals[i] = -0.20
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit long when Williams %R reaches neutral or overbought
                if wr > -50:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit short when Williams %R reaches neutral or oversold
                if wr < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_WilliamsR_1dTrend_Volume_Session"
timeframe = "1h"
leverage = 1.0
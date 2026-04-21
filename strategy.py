#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. In strong trends (price above/below 1d EMA),
# we fade extremes: short when %R > -20 in uptrend, long when %R < -80 in downtrend.
# In ranging markets (%R between -80 and -20), we wait for breakouts with volume confirmation.
# Volume > 1.5x 20-period average confirms conviction. Target: 15-30 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Williams %R (14-period)
    highest_high = prices['high'].rolling(window=14, min_periods=14).max()
    lowest_low = prices['low'].rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - prices['close']) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).fillna(-50).values
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 20-period volume average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if data not ready
        if np.isnan(williams_r[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Trend: price above/below 1d EMA34
        is_uptrend = price > ema_1d_aligned[i]
        is_downtrend = price < ema_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        if position == 0:
            if is_uptrend and volume_confirm:
                # In uptrend, fade overbought: short when Williams %R > -20
                if williams_r[i] > -20:
                    signals[i] = -0.25
                    position = -1
            elif is_downtrend and volume_confirm:
                # In downtrend, fade oversold: long when Williams %R < -80
                if williams_r[i] < -80:
                    signals[i] = 0.25
                    position = 1
            elif not is_uptrend and not is_downtrend and volume_confirm:
                # In ranging (price near EMA), wait for breakout with volume
                if price > ema_1d_aligned[i] * 1.01 and williams_r[i] > -50:
                    # Breakout above EMA with bullish momentum
                    signals[i] = 0.25
                    position = 1
                elif price < ema_1d_aligned[i] * 0.99 and williams_r[i] < -50:
                    # Breakdown below EMA with bearish momentum
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit long when overbought or trend breaks
                if williams_r[i] > -20 or price < ema_1d_aligned[i] * 0.98:
                    exit_signal = True
            elif position == -1:  # short position
                # Exit short when oversold or trend breaks
                if williams_r[i] < -80 or price > ema_1d_aligned[i] * 1.02:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsR_1dEMA34Trend_Volume"
timeframe = "12h"
leverage = 1.0
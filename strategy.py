#!/usr/bin/env python3
"""
4h_rsi_pullback_1d_trend_volume_v1
Hypothesis: On 4-hour timeframe, buy RSI pullbacks (RSI<40) in uptrends (price>1d EMA200) and sell RSI bounces (RSI>60) in downtrends (price<1d EMA200) with volume confirmation (volume>1.5x average). This captures mean reversion within the trend, works in bull (buy pullbacks) and bear (sell bounces). Target: 20-40 trades/year to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_rsi_pullback_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    d_close = df_1d['close'].values
    
    # Calculate daily EMA200 for trend filter
    daily_close_series = pd.Series(d_close)
    ema200 = daily_close_series.ewm(span=200, adjust=False).mean().values
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200)
    
    # Calculate 4h RSI (14-period)
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # Volume filter: 4h volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = vol_series / vol_ma
    vol_ratio = vol_ratio.fillna(0).values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if daily EMA not available
        if np.isnan(ema200_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine market regime based on price vs EMA200
        uptrend = close[i] > ema200_aligned[i]
        downtrend = close[i] < ema200_aligned[i]
        
        # Volume confirmation
        vol_confirmed = vol_ratio[i] > 1.5
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when RSI reaches overbought (take profit)
            if rsi[i] >= 60:
                exit_long = True
            # Exit when trend turns down
            elif not uptrend:
                exit_long = True
            # Exit when volume drops significantly
            elif vol_ratio[i] < 0.8:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit when RSI reaches oversold (take profit)
            if rsi[i] <= 40:
                exit_short = True
            # Exit when trend turns up
            elif not downtrend:
                exit_short = True
            # Exit when volume drops significantly
            elif vol_ratio[i] < 0.8:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: RSI pullback (oversold) in uptrend with volume
            long_entry = (rsi[i] < 40) and uptrend and vol_confirmed
            
            # Short entry: RSI bounce (overbought) in downtrend with volume
            short_entry = (rsi[i] > 60) and downtrend and vol_confirmed
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals
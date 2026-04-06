#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R with weekly trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions on daily chart.
# Weekly trend filter (price above/below 20-week EMA) ensures alignment with higher timeframe trend.
# Volume confirmation (current volume > 1.5x 20-day average) filters low-quality signals.
# Works in bull markets via buying oversold dips in uptrend and in bear markets via selling overbought rallies in downtrend.
# Target: 30-100 trades over 4 years (7-25/year).

name = "1d_williamsr_weekly_trend_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    williams_r = np.full(n, np.nan)
    for i in range(13, n):
        highest_high = np.max(high[i-13:i+1])
        lowest_low = np.min(low[i-13:i+1])
        if highest_high != lowest_low:
            williams_r[i] = (highest_high - close[i]) / (highest_high - lowest_low) * -100
        else:
            williams_r[i] = -50  # neutral when no range
    
    # Weekly trend filter: 20-week EMA on weekly closes
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20w = np.full(len(close_1w), np.nan)
    for i in range(19, len(close_1w)):
        if i == 19:
            ema_20w[i] = np.mean(close_1w[0:20])
        else:
            ema_20w[i] = close_1w[i] * 2/(20+1) + ema_20w[i-1] * (1 - 2/(20+1))
    ema_20w_aligned = align_htf_to_ltf(prices, df_1w, ema_20w)
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(14, n):
        # Skip if weekly trend data not available
        if np.isnan(ema_20w_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Williams %R crosses above -20 (overbought) or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.5 * atr_approx
            
            if (williams_r[i] > -20 or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Williams %R crosses below -80 (oversold) or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.5 * atr_approx
            
            if (williams_r[i] < -80 or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and weekly trend filter
            if volume_filter:
                # Buy when oversold (-80 or below) in weekly uptrend
                if (williams_r[i] <= -80 and 
                    close[i] > ema_20w_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Sell when overbought (-20 or above) in weekly downtrend
                elif (williams_r[i] >= -20 and 
                      close[i] < ema_20w_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals
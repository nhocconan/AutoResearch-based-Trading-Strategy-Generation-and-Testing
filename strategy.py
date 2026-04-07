#!/usr/bin/env python3
"""
6h_adx_volume_trend_v1
Hypothesis: On 6h timeframe, use ADX to detect strong trends (ADX>25) and enter in the trend direction when price pulls back to EMA20 with volume confirmation. Exit when ADX weakens (<20) or price moves against EMA50. This captures trend continuation during strong moves while avoiding whipsaws in ranging markets. Works in bull/bear via ADX trend filter and EMA pullback entries. Targets 15-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_adx_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA20 and EMA50 for trend and pullback
    ema20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Calculate ADX (14-period)
    # +DM, -DM, TR
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
        for i in range(period, len(data)):
            result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
        return result
    
    period = 14
    if len(plus_dm) < period:
        return np.zeros(n)
    
    smoothed_plus_dm = wilder_smooth(plus_dm, period)
    smoothed_minus_dm = wilder_smooth(minus_dm, period)
    smoothed_tr = wilder_smooth(tr, period)
    
    # Avoid division by zero
    plus_di = np.where(smoothed_tr != 0, 100 * smoothed_plus_dm / smoothed_tr, 0)
    minus_di = np.where(smoothed_tr != 0, 100 * smoothed_minus_dm / smoothed_tr, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilder_smooth(dx, period)
    
    # Align ADX to price array (add NaN at start for alignment)
    adx_full = np.concatenate([[np.nan], adx])  # prepend NaN for index 0
    
    # Volume confirmation (24-period average on 6h = 4 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(ema20[i]) or np.isnan(ema50[i]) or 
            np.isnan(adx_full[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 24-period average
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if ADX weakens (<20) - trend ending
            if adx_full[i] < 20:
                exit_long = True
            # Exit if price crosses below EMA50 (trend reversal)
            elif close[i] < ema50[i] and close[i-1] >= ema50[i-1]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if ADX weakens (<20) - trend ending
            if adx_full[i] < 20:
                exit_short = True
            # Exit if price crosses above EMA50 (trend reversal)
            elif close[i] > ema50[i] and close[i-1] <= ema50[i-1]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Strong trend condition: ADX > 25
            strong_trend = adx_full[i] > 25
            
            # Long entry: ADX>25, price pulls back to EMA20 from above, volume confirmation
            long_entry = False
            if (strong_trend and 
                close[i] >= ema20[i] and close[i-1] < ema20[i-1] and  # pullback to EMA20
                ema20[i] > ema50[i] and  # uptrend bias
                vol_confirm):
                long_entry = True
            
            # Short entry: ADX>25, price pulls back to EMA20 from below, volume confirmation
            short_entry = False
            if (strong_trend and 
                close[i] <= ema20[i] and close[i-1] > ema20[i-1] and  # pullback to EMA20
                ema20[i] < ema50[i] and  # downtrend bias
                vol_confirm):
                short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals
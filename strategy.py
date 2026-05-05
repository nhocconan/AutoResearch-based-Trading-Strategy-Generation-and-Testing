#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with volume confirmation and 12h ADX trend filter
# Long when price breaks above 12h Camarilla R3 level AND volume > 1.5x 20-period average AND 12h ADX(14) > 25
# Short when price breaks below 12h Camarilla S3 level AND volume > 1.5x 20-period average AND 12h ADX(14) > 25
# Exit when price crosses 12h Camarilla pivot point (mean reversion)
# Uses 12h primary timeframe with volume confirmation and ADX trend filter to avoid false breakouts
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe

name = "12h_Camarilla_R3S3_Breakout_Volume_ADX"
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
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Calculate 12h ADX(14) for trend filter
    # True Range
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    
    # +DM and -DM
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Wilder's smoothing function
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nanmean(data[1:period])
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
            else:
                result[i] = np.nan
        return result
    
    atr_12 = wilder_smooth(tr, 14)
    plus_di_12 = 100 * wilder_smooth(plus_dm, 14) / atr_12
    minus_di_12 = 100 * wilder_smooth(minus_dm, 14) / atr_12
    dx = np.abs(plus_di_12 - minus_di_12) / (plus_di_12 + minus_di_12) * 100
    adx_12 = wilder_smooth(dx, 14)
    trend_filter = adx_12 > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(adx_12[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 (calculated from recent 12h bar) AND volume spike AND trend filter
            # Calculate Camarilla levels from the last completed 12h bar
            if i >= 1:
                lookback_idx = i - 1
                # Ensure we have enough data for Camarilla calculation
                if lookback_idx >= 20:  # Need some history for meaningful levels
                    # Use recent 12h bar's high/low/close for Camarilla calculation
                    recent_high = high[lookback_idx]
                    recent_low = low[lookback_idx]
                    recent_close = close[lookback_idx]
                    camarilla_r3 = recent_close + (1.1 * (recent_high - recent_low) / 2)
                    camarilla_s3 = recent_close - (1.1 * (recent_high - recent_low) / 2)
                    camarilla_pivot = (recent_high + recent_low + recent_close) / 3
                    
                    if (close[i] > camarilla_r3 and 
                        volume_filter[i] and 
                        trend_filter[i]):
                        signals[i] = 0.25
                        position = 1
                    elif (close[i] < camarilla_s3 and 
                          volume_filter[i] and 
                          trend_filter[i]):
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Exit long: price crosses below Camarilla pivot (mean reversion)
            if i >= 1:
                lookback_idx = i - 1
                if lookback_idx >= 0:
                    recent_high = high[lookback_idx]
                    recent_low = low[lookback_idx]
                    recent_close = close[lookback_idx]
                    camarilla_pivot = (recent_high + recent_low + recent_close) / 3
                    
                    if close[i] < camarilla_pivot:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Camarilla pivot (mean reversion)
            if i >= 1:
                lookback_idx = i - 1
                if lookback_idx >= 0:
                    recent_high = high[lookback_idx]
                    recent_low = low[lookback_idx]
                    recent_close = close[lookback_idx]
                    camarilla_pivot = (recent_high + recent_low + recent_close) / 3
                    
                    if close[i] > camarilla_pivot:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
    
    return signals
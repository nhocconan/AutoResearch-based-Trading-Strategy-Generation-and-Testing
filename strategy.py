#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla pivot breakout with 1d volume regime filter
    # Long when price breaks above R4 + 1d volume > 1.3x 20-day average + price > 1d VWAP
    # Short when price breaks below S4 + 1d volume > 1.3x 20-day average + price < 1d VWAP
    # Exit when price returns to R3/S3 or opposite pivot level
    # Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown
    # Target: 50-150 total trades over 4 years (~12-37/year) to avoid fee drag
    # Volume regime filter ensures breakouts occur with institutional participation
    # VWAP filter ensures alignment with 1d institutional fair value
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r3 = pivot + (range_1d * 1.1 / 4)
    r4 = pivot + (range_1d * 1.1 / 2)
    s3 = pivot - (range_1d * 1.1 / 4)
    s4 = pivot - (range_1d * 1.1 / 2)
    
    # Calculate 1d VWAP (typical price * volume cumulative / volume cumulative)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    pv_1d = typical_price_1d * df_1d['volume'].values
    vol_1d = df_1d['volume'].values
    cum_pv = np.nancumsum(pv_1d)
    cum_vol = np.nancumsum(vol_1d)
    vwap_1d = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol!=0)
    
    # Calculate 1d volume average (20-period) with min_periods
    volume_1d = df_1d['volume'].values
    volume_series = pd.Series(volume_1d)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to 6h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vwap_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 1.3 * 20-period average
        vol_1d_current = df_1d['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_current)
        volume_confirmation = vol_1d_aligned[i] > 1.3 * vol_ma_aligned[i]
        
        # VWAP filter: price alignment with 1d institutional fair value
        price_above_vwap = close[i] > vwap_aligned[i]
        price_below_vwap = close[i] < vwap_aligned[i]
        
        # Breakout conditions
        bullish_breakout = (close[i] > r4_aligned[i] and 
                           volume_confirmation and 
                           price_above_vwap)
        bearish_breakout = (close[i] < s4_aligned[i] and 
                           volume_confirmation and 
                           price_below_vwap)
        
        # Exit conditions: return to R3/S3 or opposite pivot
        long_exit = close[i] < r3_aligned[i]
        short_exit = close[i] > s3_aligned[i]
        
        if bullish_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_camarilla_breakout_volume_vwap_v1"
timeframe = "6h"
leverage = 1.0
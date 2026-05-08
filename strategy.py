#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot (R4/S4) breakout with 1d volume confirmation and 1d EMA34 trend filter
# Uses daily EMA34 to filter trend direction, enters on breakout of daily Camarilla R4/S4 levels
# Volume filter requires current day volume above 20-day average to ensure participation
# Designed for 12-37 trades/year with proper risk control via trend failure
name = "12h_Camarilla_R4S4_Breakout_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Camarilla pivot, EMA34 trend, and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-day average volume for volume filter
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate daily Camarilla pivot levels
    # Camarilla: R4 = C + ((H-L) * 1.1/2), S4 = C - ((H-L) * 1.1/2)
    # where C = (H+L+C)/3 (typical price), but standard Camarilla uses previous day's H,L,C
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # First day will have NaN due to roll, handle below
    
    camarilla_r4 = prev_close + ((prev_high - prev_low) * 1.1 / 2)
    camarilla_s4 = prev_close - ((prev_high - prev_low) * 1.1 / 2)
    
    # Align 1d indicators to 12h timeframe
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Session filter: 00-23 UTC (12h bars cover full day, so always in session)
    # No session filter needed for 12h timeframe
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)  # warmup period
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get current 12h bar's OHLC for breakout detection
        curr_high = high[i]
        curr_low = low[i]
        curr_close = close[i]
        
        # Volume filter: current 1d volume must be above 20-day average
        # For 12h timeframe, we check if the most recent completed 1d bar meets volume criteria
        vol_filter = True  # Will be updated below
        
        if position == 0:
            # Look for breakout of Camarilla R4/S4 with volume confirmation
            # Long: break above R4
            if curr_high > camarilla_r4_aligned[i] and ema34_aligned[i] > 0:
                # Check volume: current 1d volume > 20-day average
                # Find the most recent completed 1d bar
                idx_1d = len(df_1d) - 1
                while idx_1d >= 0 and df_1d.iloc[idx_1d]['open_time'] > prices.iloc[i]['open_time']:
                    idx_1d -= 1
                if idx_1d >= 0:
                    vol_1d_current = df_1d.iloc[idx_1d]['volume']
                    vol_filter = vol_1d_current > vol_ma_20_aligned[i] if vol_ma_20_aligned[i] > 0 else False
                else:
                    vol_filter = False
                
                if vol_filter:
                    signals[i] = 0.25
                    position = 1
            # Short: break below S4
            elif curr_low < camarilla_s4_aligned[i] and ema34_aligned[i] > 0:
                # Check volume: current 1d volume > 20-day average
                idx_1d = len(df_1d) - 1
                while idx_1d >= 0 and df_1d.iloc[idx_1d]['open_time'] > prices.iloc[i]['open_time']:
                    idx_1d -= 1
                if idx_1d >= 0:
                    vol_1d_current = df_1d.iloc[idx_1d]['volume']
                    vol_filter = vol_1d_current > vol_ma_20_aligned[i] if vol_ma_20_aligned[i] > 0 else False
                else:
                    vol_filter = False
                
                if vol_filter:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price breaks back below S4 (failed breakout) or trend changes
            if curr_low < camarilla_s4_aligned[i] or ema34_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks back above R4 (failed breakdown) or trend changes
            if curr_high > camarilla_r4_aligned[i] or ema34_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
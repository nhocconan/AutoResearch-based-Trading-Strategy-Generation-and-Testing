#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Weekly Camarilla Pivot Reversal with Volume Confirmation
# Long when price touches S1/S2 with reversal signal + volume spike + weekly uptrend
# Short when price touches R1/R2 with reversal signal + volume spike + weekly downtrend
# Uses weekly trend filter and daily Camarilla pivots for low-frequency, high-conviction trades
# Designed for 15-25 trades/year to avoid fee drag while capturing reversal moves in all market regimes

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # Calculate 34-period EMA on weekly close for trend filter
    ema_34_weekly = pd.Series(close_weekly).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_weekly, ema_34_weekly)
    
    # Load daily data for Camarilla pivot calculation
    df_daily = get_htf_data(prices, '1d')
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate Camarilla pivot levels (R1, R2, S1, S2) from previous day
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R1 = C + (Range * 1.1 / 12)
    # R2 = C + (Range * 1.1 / 6)
    # S1 = C - (Range * 1.1 / 12)
    # S2 = C - (Range * 1.1 / 6)
    pivot_daily = (high_daily + low_daily + close_daily) / 3
    range_daily = high_daily - low_daily
    r1_daily = close_daily + (range_daily * 1.1 / 12)
    r2_daily = close_daily + (range_daily * 1.1 / 6)
    s1_daily = close_daily - (range_daily * 1.1 / 12)
    s2_daily = close_daily - (range_daily * 1.1 / 6)
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_daily, r1_daily)
    r2_aligned = align_htf_to_ltf(prices, df_daily, r2_daily)
    s1_aligned = align_htf_to_ltf(prices, df_daily, s1_daily)
    s2_aligned = align_htf_to_ltf(prices, df_daily, s2_daily)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(s2_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        r1 = r1_aligned[i]
        r2 = r2_aligned[i]
        s1 = s1_aligned[i]
        s2 = s2_aligned[i]
        ema_val = ema_34_aligned[i]
        
        # Volume filter: current volume > 2.2 * 20-period average
        vol_spike = vol > 2.2 * vol_ma
        
        # Price action signals: check for rejection at pivot levels
        # Bullish rejection: long wick, close > open
        # Bearish rejection: long wick, close < open
        open_price = prices['open'].iloc[i]
        close_price = prices['close'].iloc[i]
        high_price = prices['high'].iloc[i]
        low_price = prices['low'].iloc[i]
        
        body_size = abs(close_price - open_price)
        upper_wick = high_price - max(open_price, close_price)
        lower_wick = min(open_price, close_price) - low_price
        
        bullish_rejection = (lower_wick > body_size * 1.5) and (close_price > open_price)
        bearish_rejection = (upper_wick > body_size * 1.5) and (close_price < open_price)
        
        if position == 0:
            # Long conditions: price touches S1/S2 with bullish rejection + weekly uptrend + volume spike
            if ((abs(price - s1) < (r2 - s2) * 0.02) or (abs(price - s2) < (r2 - s2) * 0.02)) and \
               bullish_rejection and \
               price > ema_val and \
               vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price touches R1/R2 with bearish rejection + weekly downtrend + volume spike
            elif ((abs(price - r1) < (r2 - s2) * 0.02) or (abs(price - r2) < (r2 - s2) * 0.02)) and \
                 bearish_rejection and \
                 price < ema_val and \
                 vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price moves against position or reaches opposite pivot level
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price reaches R1 or shows bearish rejection at resistance
                if price >= r1 or bearish_rejection:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price reaches S1 or shows bullish rejection at support
                if price <= s1 or bullish_rejection:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WeeklyTrend_CamarillaPivot_Reversal"
timeframe = "12h"
leverage = 1.0
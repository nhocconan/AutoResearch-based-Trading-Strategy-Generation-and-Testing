#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R4/S4 breakout with 1w EMA21 trend filter and volume spike confirmation
# Camarilla R4/S4 levels represent extreme support/resistance, reducing false breakouts
# 1w EMA21 ensures alignment with weekly trend to avoid counter-trend whipsaws
# Volume spike (2.0x 50-period average) confirms institutional participation
# Discrete sizing 0.25 minimizes fee churn. Target: 30-100 total trades over 4 years (7-25/year).
# Works in bull markets via breakouts above R4 and bear markets via breakdowns below S4 with trend filter.

name = "1d_Camarilla_R4S4_1wEMA21_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1w data ONCE before loop (MTF Rule #1)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA21 for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate 1d Camarilla pivot levels (R4, S4)
    high_1d = df_1w['high'].values  # Use weekly high/low for Camarilla calculation
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla levels based on prior week's OHLC (R4/S4)
    camarilla_r4 = close_1w + ((high_1w - low_1w) * 1.5 / 2)
    camarilla_s4 = close_1w - ((high_1w - low_1w) * 1.5 / 2)
    
    # Align Camarilla levels to 1d timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    # Volume confirmation: volume > 2.0x 50-period average (50*1d = 50 days)
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (2.0 * vol_ma_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 50)  # warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(vol_ma_50[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_21_1w = ema_21_1w_aligned[i]
        curr_r4 = camarilla_r4_aligned[i]
        curr_s4 = camarilla_s4_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and price above/below EMA21_1w for trend alignment
            if curr_volume_spike:
                # Bullish entry: break above R4 with price > EMA21_1w
                if curr_close > curr_r4 and curr_close > curr_ema_21_1w:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: break below S4 with price < EMA21_1w
                elif curr_close < curr_s4 and curr_close < curr_ema_21_1w:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below R4 (breakout fails) OR price crosses below EMA21_1w
            if curr_close < curr_r4 or curr_close < curr_ema_21_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above S4 (breakdown fails) OR price crosses above EMA21_1w
            if curr_close > curr_s4 or curr_close > curr_ema_21_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
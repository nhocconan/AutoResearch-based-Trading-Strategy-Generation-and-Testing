#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike
# Camarilla pivots identify key intraday support/resistance levels (R1, S1)
# Breakout above R1 or below S1 with volume confirmation signals institutional participation
# 4h EMA50 filter ensures alignment with intermediate trend to avoid counter-trend trades
# Session filter (08-20 UTC) reduces noise during low-liquidity hours
# Works in bull/bear: volume confirms breakout validity, 4h EMA50 filters whipsaws
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe

name = "1h_Camarilla_R1S1_VolumeSpike_4hEMA50_Trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate daily Camarilla pivots for R1 and S1 levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Camarilla formulas: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_r1 = df_1d['close'] + (1.1 * (df_1d['high'] - df_1d['low']) / 12)
    camarilla_s1 = df_1d['close'] - (1.1 * (df_1d['high'] - df_1d['low']) / 12)
    r1_4h = align_htf_to_ltf(prices, df_1d, camarilla_r1.values)
    s1_4h = align_htf_to_ltf(prices, df_1d, camarilla_s1.values)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # Session filter: 08-20 UTC (avoid low-liquidity hours)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20)  # warmup for 4h EMA50 and volume MA20
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(r1_4h[i]) or np.isnan(s1_4h[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        curr_session = session_filter[i]
        curr_ema_50 = ema_50_4h_aligned[i]
        curr_r1 = r1_4h[i]
        curr_s1 = s1_4h[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation, session filter, and trend filter
            if curr_volume_confirm and curr_session:
                # Bullish entry: break above R1 with close above 4h EMA50
                if curr_close > curr_r1 and curr_close > curr_ema_50:
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
                # Bearish entry: break below S1 with close below 4h EMA50
                elif curr_close < curr_s1 and curr_close < curr_ema_50:
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price breaks below S1 (invalidates bullish breakout)
            if curr_close < curr_s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit when price breaks above R1 (invalidates bearish breakout)
            if curr_close > curr_r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
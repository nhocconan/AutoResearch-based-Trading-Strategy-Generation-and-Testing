#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h data (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # 6h Donchian(15) for breakout detection
    high_15_6h = pd.Series(high_6h).rolling(window=15, min_periods=15).max().values
    low_15_6h = pd.Series(low_6h).rolling(window=15, min_periods=15).min().values
    donchian_upper_6h = align_htf_to_ltf(prices, df_6h, high_15_6h)
    donchian_lower_6h = align_htf_to_ltf(prices, df_6h, low_15_6h)
    
    # === 1d data (HTF for pivot levels) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily pivot points (standard)
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    r1_1d = (2 * pivot_1d) - low_1d
    s1_1d = (2 * pivot_1d) - high_1d
    r2_1d = pivot_1d + (high_1d - low_1d)
    s2_1d = pivot_1d - (high_1d - low_1d)
    r3_1d = high_1d + 2 * (pivot_1d - low_1d)
    s3_1d = low_1d - 2 * (high_1d - pivot_1d)
    
    # Align daily pivot levels to 6t chart
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # === 1d data for volume confirmation ===
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # === 6h indicators ===
    # RSI(14) for overbought/oversold
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume ratio (current vs 20-period MA)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_6h[i]) or np.isnan(donchian_lower_6h[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = close[i]
        upper_6h = donchian_upper_6h[i]
        lower_6h = donchian_lower_6h[i]
        rsi_val = rsi[i]
        vol_ratio_val = vol_ratio[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        r2 = r2_1d_aligned[i]
        s2 = s2_1d_aligned[i]
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        vol_ma_20_1d_val = vol_ma_20_1d_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below S1 or RSI becomes overbought
            if (price < s1) or (rsi_val > 70):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above R1 or RSI becomes oversold
            if (price > r1) or (rsi_val < 30):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Only trade during session
            if in_session:
                # LONG: Price breaks above R1 AND above R2 (strong breakout)
                # AND RSI not overbought AND volume spike above daily average
                if (price > r1) and (price > r2) and (rsi_val < 60) and \
                   (vol_ratio_val > 1.5) and (volume > vol_ma_20_1d_val):
                    signals[i] = 0.25
                    position = 1
                    continue
                
                # SHORT: Price breaks below S1 AND below S2 (strong breakdown)
                # AND RSI not oversold AND volume spike above daily average
                elif (price < s1) and (price < s2) and (rsi_val > 40) and \
                     (vol_ratio_val > 1.5) and (volume > vol_ma_20_1d_val):
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Pivot_R1_R2_Breakout_Volume_RSI"
timeframe = "6h"
leverage = 1.0
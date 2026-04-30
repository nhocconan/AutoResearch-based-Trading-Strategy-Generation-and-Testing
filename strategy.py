#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation.
# In trending markets (price > 4h EMA50), break above R1 or below S1 with volume triggers continuation entries.
# In ranging markets (price near 4h EMA50), fade at extreme R4/S4 levels for mean reversion.
# Session filter (08-20 UTC) reduces noise trades. Size: 0.20. Target: 15-37 trades/year.

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for Camarilla pivot levels and EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h Camarilla pivot levels (R1, S1, R4, S4)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels
    camarilla_r1 = close_4h + 1.1 * (high_4h - low_4h) / 12
    camarilla_s1 = close_4h - 1.1 * (high_4h - low_4h) / 12
    camarilla_r4 = close_4h + 1.1 * (high_4h - low_4h)
    camarilla_s4 = close_4h - 1.1 * (high_4h - low_4h)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h Camarilla levels and EMA50 to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    r4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s4)
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: volume > 2.0x 24-period average (strict to reduce trades)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if outside session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Regime filter: price above/below 4h EMA50 determines trend direction
        is_uptrend = close[i] > ema_50_aligned[i]
        is_downtrend = close[i] < ema_50_aligned[i]
        
        curr_close = close[i]
        curr_r1 = r1_aligned[i]
        curr_s1 = s1_aligned[i]
        curr_r4 = r4_aligned[i]
        curr_s4 = s4_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            if is_uptrend:
                # In uptrend: look for long breakouts above R1 with volume
                if curr_close > curr_r1 and curr_volume_spike:
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
            elif is_downtrend:
                # In downtrend: look for short breakdowns below S1 with volume
                if curr_close < curr_s1 and curr_volume_spike:
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
            else:
                # In ranging market (near EMA): mean reversion at extreme Camarilla levels
                if curr_close < curr_s4:
                    # Deep oversold: look for long
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
                elif curr_close > curr_r4:
                    # Deep overbought: look for short
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit on close below R1 (mean reversion) or opposite volume spike
            if curr_close < curr_r1 or (not is_uptrend and curr_volume_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit on close above S1 (mean reversion) or opposite volume spike
            if curr_close > curr_s1 or (not is_downtrend and curr_volume_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
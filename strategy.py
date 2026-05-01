#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume spike confirmation.
# Camarilla pivot levels from 1d data provide strong intraday support/resistance.
# Breakout above R3 or below S3 with volume confirmation indicates strong momentum.
# 1w EMA50 filter ensures we only trade in the direction of the weekly trend.
# Volume spike (current 4h volume > 2.0x 20-bar average) confirms institutional participation.
# Designed for low trade frequency (<50/year) to minimize fee drag and work in both bull/bear markets.

name = "4h_Camarilla_R3S3_Breakout_1wEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if np.isnan(ema_50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        
        # Calculate Camarilla pivot levels from previous 1d bar
        # Need to get the previous completed 1d bar's OHLC
        # Since we're in 4h timeframe, we need to map to 1d index
        # Use align_htf_to_ltf to get the previous 1d bar's values aligned to 4h
        if i < start_idx:
            signals[i] = 0.0
            continue
            
        # Get previous 1d bar's OHLC (aligned to current 4h bar)
        # We need to use the 1d bar that closed before the current 4h bar
        # align_htf_to_ltf with proper alignment will give us the previous completed 1d bar's values
        # For simplicity, we'll calculate Camarilla levels using the 1d data and align them
        
        # Calculate Camarilla levels for each 1d bar
        # Camarilla levels: 
        # H4 = Close + 1.1*(High-Low)/2
        # L4 = Close - 1.1*(High-Low)/2
        # H3 = Close + 1.1*(High-Low)/4
        # L3 = Close - 1.1*(High-Low)/4
        # H2 = Close + 1.1*(High-Low)/6
        # L2 = Close - 1.1*(High-Low)/6
        # H1 = Close + 1.1*(High-Low)/12
        # L1 = Close - 1.1*(High-Low)/12
        # R3 = H3, S3 = L3 (we'll use H3/L3 as our breakout levels)
        
        # We'll calculate these for the 1d data and then align
        # But to avoid look-ahead, we need to use the previous 1d bar's data
        
        # Instead, let's calculate the Camarilla levels using a rolling window on 1d data
        # and then align to 4h
        
        # For now, we'll use a simplified approach: calculate typical price and use ATR-like measure
        # But to properly implement Camarilla, we need the previous day's OHLC
        
        # Given the complexity and to avoid look-ahead issues, let's use a different approach:
        # Use Donchian breakout with 20-period on 4h as a proxy for Camarilla breakout
        # This is a simplification but avoids the MTF alignment complexity for pivot levels
        
        # Actually, let's properly implement this by getting the previous 1d bar's OHLC
        # We can do this by indexing into the 1d data
        
        # Calculate how many 4h bars in a 1d bar: 24h / 4h = 6
        bars_per_1d = 6
        
        # Get the index of the previous 1d bar in 1d data
        # Current 4h bar index i corresponds to 1d bar index: i // bars_per_1d
        # We want the previous 1d bar: (i // bars_per_1d) - 1
        idx_1d = (i // bars_per_1d) - 1
        
        if idx_1d < 0:
            signals[i] = 0.0
            continue
            
        # Get previous 1d bar's OHLC
        if idx_1d >= len(df_1d):
            signals[i] = 0.0
            continue
            
        prev_high = df_1d['high'].iloc[idx_1d]
        prev_low = df_1d['low'].iloc[idx_1d]
        prev_close = df_1d['close'].iloc[idx_1d]
        
        # Calculate Camarilla levels
        range_hl = prev_high - prev_low
        if range_hl <= 0:
            signals[i] = 0.0
            continue
            
        # Camarilla H3 and L3 (our breakout levels)
        h3 = prev_close + 1.1 * range_hl / 4
        l3 = prev_close - 1.1 * range_hl / 4
        
        # Volume confirmation: current 4h volume > 2.0x 20-period average
        # We'll calculate volume MA on 4h data
        if i < 20:
            vol_ma = 0
        else:
            vol_ma = np.mean(volume[i-20:i])
        volume_confirm = curr_vol > (vol_ma * 2.0)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: break above H3 AND price > 1w EMA50 AND volume confirmation
            if (curr_close > h3 and 
                curr_close > curr_ema_50_1w and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: break below L3 AND price < 1w EMA50 AND volume confirmation
            elif (curr_close < l3 and 
                  curr_close < curr_ema_50_1w and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < 1w EMA50 (trend violation) OR reverse signal
            if (curr_close < curr_ema_50_1w or 
                curr_close < l3):  # reverse break below L3
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price > 1w EMA50 (trend violation) OR reverse signal
            if (curr_close > curr_ema_50_1w or 
                curr_close > h3):  # reverse break above H3
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
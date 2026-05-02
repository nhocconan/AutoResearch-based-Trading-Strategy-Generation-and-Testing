#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses 6h timeframe for signal generation with Camarilla R3/S3 breakouts
# 1d EMA34 provides multi-timeframe trend filter to avoid counter-trend trades
# Volume confirmation (2.0x 20-period average) ensures institutional participation
# Discrete position sizing (0.25) minimizes fee churn
# Target: 50-150 total trades over 4 years = 12-37/year for 6h timeframe
# Works in bull markets via trend-aligned breakouts, in bear via volume confirmation filtering weak moves
# Based on proven Camarilla pivot strategy family with proper 6h implementation

name = "6h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values  # Camarilla uses open of the period
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar (need 1d OHLC)
    # We'll calculate these for each 6h bar using the most recent completed 1d bar
    # For simplicity, we use the current day's OHLC to calculate levels (standard approach)
    # In practice, Camarilla uses previous day's OHLC, but for intraday we use session OHLC
    
    # Calculate typical price for the day (we'll use rolling window to get daily OHLC)
    # Since we're on 6h timeframe, we need to aggregate to 1d for Camarilla calculation
    # But per rules, we must use get_htf_data for actual 1d data
    
    # Get 1d OHLC from the actual 1d data
    # We need to align the 1d OHLC to each 6h bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Calculate Camarilla levels for each 1d bar
    # Camarilla levels:
    # R4 = close + ((high - low) * 1.1 / 2)
    # R3 = close + ((high - low) * 1.1 / 4)
    # R2 = close + ((high - low) * 1.1 / 6)
    # R1 = close + ((high - low) * 1.1 / 12)
    # PP = (high + low + close) / 3
    # S1 = close - ((high - low) * 1.1 / 12)
    # S2 = close - ((high - low) * 1.1 / 6)
    # S3 = close - ((high - low) * 1.1 / 4)
    # S4 = close - ((high - low) * 1.1 / 2)
    
    rng = high_1d - low_1d
    r4 = close_1d + (rng * 1.1 / 2)
    r3 = close_1d + (rng * 1.1 / 4)
    r2 = close_1d + (rng * 1.1 / 6)
    r1 = close_1d + (rng * 1.1 / 12)
    pp = (high_1d + low_1d + close_1d) / 3
    s1 = close_1d - (rng * 1.1 / 12)
    s2 = close_1d - (rng * 1.1 / 6)
    s3 = close_1d - (rng * 1.1 / 4)
    s4 = close_1d - (rng * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above R3 with volume confirmation and price > 1d EMA34
            if close[i] > r3_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with volume confirmation and price < 1d EMA34
            elif close[i] < s3_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below R1 (reversal) or S1 (strong reversal)
            if close[i] < r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above S1 (reversal) or R1 (strong reversal)
            if close[i] > s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
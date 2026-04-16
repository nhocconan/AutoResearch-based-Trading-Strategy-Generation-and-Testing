#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla pivot breakout with volume confirmation and 1d trend filter.
# Long when price breaks above R1 with volume > 1.5x average volume AND price > 1d EMA34.
# Short when price breaks below S1 with volume > 1.5x average volume AND price < 1d EMA34.
# Exit when price retouches the pivot point (PP).
# Uses 4h for signal direction (Camarilla levels), 1d for trend filter (EMA34), 1h for entry timing.
# Session filter: 08-20 UTC to avoid low-liquidity hours.
# Position size: 0.20 discrete to minimize fee churn.
# Target: 15-35 trades/year per symbol to stay within fee limits.
# Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once before loop for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Get 1d data once before loop for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === 4h Indicators: Camarilla Pivot Points (R1, S1, PP) ===
    # Typical Price = (high + low + close) / 3
    typical_price_4h = (high_4h + low_4h + close_4h) / 3.0
    # Pivot Point (PP) = typical_price_4h (for Camarilla, PP is the typical price)
    pp_4h = typical_price_4h
    # Range = high_4h - low_4h
    range_4h = high_4h - low_4h
    # Resistance 1 (R1) = PP + (range * 1.1/12)
    r1_4h = pp_4h + (range_4h * 1.1 / 12)
    # Support 1 (S1) = PP - (range * 1.1/12)
    s1_4h = pp_4h - (range_4h * 1.1 / 12)
    
    # === 1d Indicators: EMA34 for trend filter ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === 1h Indicators: Average Volume (20-period) for volume confirmation ===
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary timeframe (1h)
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    pp_aligned = align_htf_to_ltf(prices, df_4h, pp_4h)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    volume_sma_aligned = align_htf_to_ltf(prices, df_4h, volume_sma_20)  # align 4h volume avg to 1h
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100  # 20-period volume SMA + 34-period EMA + 4h data alignment
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(ema34_aligned[i]) or
            np.isnan(volume_sma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        pp = pp_aligned[i]
        ema34 = ema34_aligned[i]
        vol_avg = volume_sma_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price retouches pivot point (PP) or breaks below S1 (failed breakout)
            if (price <= pp) or (price < s1):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price retouches pivot point (PP) or breaks above R1 (failed breakdown)
            if (price >= pp) or (price > r1):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume confirmation: current volume > 1.5x average volume
            volume_confirm = vol > (1.5 * vol_avg)
            
            # LONG: Price breaks above R1 with volume confirmation AND price > 1d EMA34 (uptrend)
            if (price > r1) and volume_confirm and (price > ema34):
                signals[i] = 0.20
                position = 1
            
            # SHORT: Price breaks below S1 with volume confirmation AND price < 1d EMA34 (downtrend)
            elif (price < s1) and volume_confirm and (price < ema34):
                signals[i] = -0.20
                position = -1
        
        else:
            signals[i] = position * 0.20  # maintain position
    
    return signals

name = "1h_4hCamarilla_R1S1_VolumeConfirm_1dEMA34_TrendFilter_V1"
timeframe = "1h"
leverage = 1.0
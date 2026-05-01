#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + Elder Ray + Volume Spike
# Uses 1d timeframe with weekly trend filter (1w EMA50) to avoid counter-trend trades
# Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and strength
# Elder Ray (Bull/Bear Power) confirms momentum with EMA13
# Volume spike filters false breakouts
# Designed for very low frequency (<30 trades/year) to minimize fee drag
# Works in bull/bear via trend filter + momentum confirmation

name = "1d_WilliamsAlligator_ElderRay_VolumeSpike_v1"
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
    
    # 1d data for Williams Alligator and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1w HTF data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter (major trend direction)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator: SMAs of median price
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars
    # Lips: 5-period SMMA shifted 3 bars
    median_price = (high + low) / 2.0
    
    # Calculate SMMA (Smoothed Moving Average) - similar to Wilder's smoothing
    def smma(source, period):
        if len(source) < period:
            return np.full_like(source, np.nan)
        result = np.full_like(source, np.nan)
        # First value is SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CURRENT) / PERIOD
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    # Alligator lines
    jaw = smma(median_price, 13)  # Jaw (Blue)
    teeth = smma(median_price, 8)  # Teeth (Red)
    lips = smma(median_price, 5)   # Lips (Green)
    
    # Shift Alligator lines (Jaw: 8, Teeth: 5, Lips: 3)
    jaw_shifted = np.concatenate([[np.nan]*8, jaw[:-8]]) if len(jaw) > 8 else np.full_like(jaw, np.nan)
    teeth_shifted = np.concatenate([[np.nan]*5, teeth[:-5]]) if len(teeth) > 5 else np.full_like(teeth, np.nan)
    lips_shifted = np.concatenate([[np.nan]*3, lips[:-3]]) if len(lips) > 3 else np.full_like(lips, np.nan)
    
    # Align Alligator to 1d timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_shifted)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, 34, 20)  # Need 1w EMA50, Alligator, EMA13, volume MA20
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Alligator trend conditions
        # Uptrend: Lips > Teeth > Jaw (Green > Red > Blue)
        alligator_uptrend = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        # Downtrend: Lips < Teeth < Jaw (Green < Red < Blue)
        alligator_downtrend = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Elder Ray conditions
        strong_bull = bull_power[i] > 0 and bull_power[i] > np.mean(bull_power[max(0, i-10):i+1])  # Above average
        strong_bear = bear_power[i] < 0 and bear_power[i] < np.mean(bear_power[max(0, i-10):i+1])  # Below average
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Alligator uptrend + strong bull power + volume spike
            if alligator_uptrend and strong_bull and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Alligator downtrend + strong bear power + volume spike
            elif alligator_downtrend and strong_bear and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Alligator trend reversal or weak bull power
            if not alligator_uptrend or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Alligator trend reversal or weak bear power
            if not alligator_downtrend or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
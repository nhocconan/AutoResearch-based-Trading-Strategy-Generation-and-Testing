#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray combination with 1w trend filter and volume confirmation
# Uses 12h primary timeframe for lower trade frequency (~25-40/year) to minimize fee drag
# Williams Alligator (jaw/teeth/lips) identifies trend state and potential reversals
# Elder Ray (Bull Power/Bear Power) measures trend strength via EMA13 deviation
# 1w EMA34 trend filter ensures alignment with weekly momentum for higher reliability
# Volume spike (1.8x 30-period average) confirms institutional participation
# Designed for both bull and bear markets: Alligator catches trends, Elder Ray filters weak moves
# Tight entry conditions target 50-120 total trades over 4 years (12-30/year) to avoid overtrading

name = "12h_WilliamsAlligator_ElderRay_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for HTF calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Williams Alligator on 12h data (smoothed with 5-period SMA)
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    def smma(data, period):
        """Smoothed Moving Average"""
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_DATA) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(close, 13)  # Jaw (blue)
    teeth = smma(close, 8)  # Teeth (red)
    lips = smma(close, 5)   # Lips (green)
    
    # Calculate Elder Ray on 12h data
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    # Calculate volume spike (1.8x 30-period average)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Alligator, Elder Ray, and volume MA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
            # Elder Ray confirmation: Bull Power > 0 and rising for long, Bear Power < 0 and falling for short
            # Volume spike for institutional confirmation
            
            # Long: Alligator bullish alignment + Bull Power positive + price > 1w EMA34 + volume spike
            if (lips[i] > teeth[i] > jaw[i] and 
                bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and
                close[i] > ema_34_1w_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish alignment + Bear Power negative + price < 1w EMA34 + volume spike
            elif (lips[i] < teeth[i] < jaw[i] and 
                  bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and
                  close[i] < ema_34_1w_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator loses bullish alignment OR Bear Power turns positive (trend weakness)
            if not (lips[i] > teeth[i] > jaw[i]) or bear_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator loses bearish alignment OR Bull Power turns negative (trend weakness)
            if not (lips[i] < teeth[i] < jaw[i]) or bull_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
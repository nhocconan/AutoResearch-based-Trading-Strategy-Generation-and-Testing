#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator (SMMA) trend filter with weekly EMA21 trend confirmation and volume spike.
# The Williams Alligator uses three smoothed moving averages (Jaw:13, Teeth:8, Lips:5) to identify trends.
# Long when Lips > Teeth > Jaw (bullish alignment) with price above weekly EMA21 and volume spike.
# Short when Lips < Teeth < Jaw (bearish alignment) with price below weekly EMA21 and volume spike.
# Weekly EMA21 provides higher timeframe trend filter to avoid counter-trend trades.
# Designed for low trade frequency (~10-25/year) on daily timeframe to minimize fee decay.
# Works in both bull and bear markets by following higher timeframe trend and requiring volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Williams Alligator calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator SMMA (Smoothed Moving Average)
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    def smma(series, period):
        # Smoothed Moving Average: similar to EMA but with smoothing
        sma = pd.Series(series).rolling(window=period, min_periods=period).mean().values
        # Apply smoothing: SMMA today = (SMMA yesterday * (period-1) + price today) / period
        smma_vals = np.full_like(series, np.nan, dtype=float)
        if len(sma) >= period:
            smma_vals[period-1] = sma[period-1]  # First value is SMA
            for i in range(period, len(series)):
                if not np.isnan(sma[i]):
                    smma_vals[i] = (smma_vals[i-1] * (period-1) + sma[i]) / period
        return smma_vals
    
    jaw = smma(close_1d, 13)   # Jaw (blue line)
    teeth = smma(close_1d, 8)  # Teeth (red line)
    lips = smma(close_1d, 5)   # Lips (green line)
    
    # Load weekly data for EMA21 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Align 1d indicators to daily timeframe (waits for 1d bar to close)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema_21w_val = ema_21_1w_aligned[i]
        
        # Volume filter: current volume > 1.8 * 20-period average (moderate filter)
        vol_spike = vol > 1.8 * vol_ma
        
        # Williams Alligator signals
        bullish_alignment = lips_val > teeth_val > jaw_val  # Lips > Teeth > Jaw
        bearish_alignment = lips_val < teeth_val < jaw_val  # Lips < Teeth < Jaw
        
        if position == 0:
            # Long conditions: bullish alignment + price above weekly EMA21 + volume spike
            if bullish_alignment and price > ema_21w_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish alignment + price below weekly EMA21 + volume spike
            elif bearish_alignment and price < ema_21w_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when bearish alignment or price breaks below weekly EMA21
                if bearish_alignment or price < ema_21w_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when bullish alignment or price breaks above weekly EMA21
                if bullish_alignment or price > ema_21w_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WilliamsAlligator_1wEMA21_Volume"
timeframe = "1d"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator system with 1d trend filter and volume confirmation
# Uses Williams Alligator (Jaw, Teeth, Lips) to identify trend direction and alignment
# Long when Lips > Teeth > Jaw (bullish alignment) + price above Teeth + 1d uptrend + volume spike
# Short when Lips < Teeth < Jaw (bearish alignment) + price below Teeth + 1d downtrend + volume spike
# Exit when Alligator alignment breaks or price crosses Jaw
# Designed for low trade frequency (~15-30/year on 12h) to minimize fee drain. Works in trending markets.
# Williams Alligator is effective in catching and riding trends while avoiding choppy markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 34-period EMA on 1d close for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator on 12h data (5, 8, 13 periods shifted)
    # Jaw (blue line): 13-period SMMA, shifted 8 bars
    # Teeth (red line): 8-period SMMA, shifted 5 bars
    # Lips (green line): 5-period SMMA, shifted 3 bars
    close = prices['close'].values
    
    def smma(data, period):
        """Smoothed Moving Average"""
        sma = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            sma[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                sma[i] = (sma[i-1] * (period-1) + data[i]) / period
        return sma
    
    # Calculate SMMA values
    jaw_raw = smma(close, 13)
    teeth_raw = smma(close, 8)
    lips_raw = smma(close, 5)
    
    # Apply shifts (Jaw: +8, Teeth: +5, Lips: +3)
    jaw = np.full_like(jaw_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    if len(jaw_raw) > 8:
        jaw[8:] = jaw_raw[:-8]
    if len(teeth_raw) > 5:
        teeth[5:] = teeth_raw[:-5]
    if len(lips_raw) > 3:
        lips[3:] = lips_raw[:-3]
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        ema_val = ema_34_aligned[i]
        
        # Volume filter: current volume > 1.8 * 20-period average
        vol_spike = vol > 1.8 * vol_ma
        
        # Alligator alignment checks
        bullish_alignment = lips_val > teeth_val > jaw_val
        bearish_alignment = lips_val < teeth_val < jaw_val
        
        if position == 0:
            # Long conditions: bullish alignment + price above teeth + uptrend + volume spike
            if bullish_alignment and price > teeth_val and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish alignment + price below teeth + downtrend + volume spike
            elif bearish_alignment and price < teeth_val and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: alignment breaks or price crosses jaw
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when bullish alignment breaks or price crosses below jaw
                if not bullish_alignment or price < jaw_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when bearish alignment breaks or price crosses above jaw
                if not bearish_alignment or price > jaw_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0
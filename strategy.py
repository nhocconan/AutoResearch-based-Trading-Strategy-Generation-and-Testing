#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6s Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Long when Alligator Jaw < Teeth < Lips (bullish alignment) AND price > 1d EMA50 AND volume > 1.5x 20-period average volume
# Short when Alligator Jaw > Teeth > Lips (bearish alignment) AND price < 1d EMA50 AND volume > 1.5x 20-period average volume
# ATR trailing stop (2.0x ATR) to manage risk
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag on 6h timeframe
# Williams Alligator identifies trend alignment, 1d EMA50 filter avoids counter-trend trades, volume confirmation adds conviction

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d EMA50 trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 6h Williams Alligator (13,8,5) ===
    # Jaw: 13-period SMMA (smoothed moving average) of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price = (high + low) / 2
    
    # SMMA calculation (similar to Wilder's smoothing)
    def smma(source, period):
        result = np.full_like(source, np.nan, dtype=float)
        if len(source) >= period:
            # First value is simple average
            result[period-1] = np.mean(source[:period])
            # Subsequent values: (prev*(period-1) + current) / period
            for i in range(period, len(source)):
                result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw_raw = smma(median_price, 13)
    teeth_raw = smma(median_price, 8)
    lips_raw = smma(median_price, 5)
    
    # Shift the averages (Jaw: 8, Teeth: 5, Lips: 3)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Invalidate the shifted values
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # === 6h Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 6h ATR for trailing stop (14-period) ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_1d_aligned[i]) or 
            np.isnan(jaw[i]) or
            np.isnan(teeth[i]) or
            np.isnan(lips[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_val = ema_1d_aligned[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        vol_confirm = volume[i] > vol_ma_20[i] * 1.5  # 1.5x average volume for confirmation
        atr_val = atr[i]
        
        # === TRAILING STOP LOGIC ===
        if position == 1:  # Long position
            # Update highest price since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Trail stop: exit if price drops 2.0*ATR from highest
            if atr_val > 0 and price < highest_since_entry - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            if price < lowest_since_entry or lowest_since_entry == 0:
                lowest_since_entry = price
            # Trail stop: exit if price rises 2.0*ATR from lowest
            if atr_val > 0 and price > lowest_since_entry + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Bullish alignment: Jaw < Teeth < Lips
            bullish_alignment = jaw_val < teeth_val and teeth_val < lips_val
            # Bearish alignment: Jaw > Teeth > Lips
            bearish_alignment = jaw_val > teeth_val and teeth_val > lips_val
            
            # Long when: bullish alignment AND price > EMA50 AND volume confirmation
            if bullish_alignment and price > ema_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
                continue
            # Short when: bearish alignment AND price < EMA50 AND volume confirmation
            elif bearish_alignment and price < ema_val and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6s_WilliamsAlligator_1dEMA50_Volume1.5x_ATRTrail"
timeframe = "6h"
leverage = 1.0
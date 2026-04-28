#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + Williams Alligator combination with volume confirmation
# Long when: ADX(14) > 25 (trending) + Alligator Jaw < Teeth < Lips (bullish alignment) + close > Alligator Lips + volume > 1.5x 20-bar avg
# Short when: ADX(14) > 25 (trending) + Alligator Jaw > Teeth > Lips (bearish alignment) + close < Alligator Lips + volume > 1.5x 20-bar avg
# Exit when: ADX < 20 (trend weak) OR Alligator lines cross (Jaw-Teeth or Teeth-Lips)
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year on 6h.
# Works in bull markets by catching strong trends, works in bear by requiring volume spikes which often accompany panic selling/buying climaxes that precede reversals.
# Williams Alligator uses SMAs: Jaw (13,8), Teeth (8,5), Lips (5,3) - all shifted forward by respective amounts.

name = "6h_WilliamsAlligator_ADX_Trend_VolumeFilter_v1"
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
    
    # Calculate Williams Alligator components
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward  
    # Lips: 5-period SMMA, shifted 3 bars forward
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    
    def smma(data, period):
        """Smoothed Moving Average"""
        if len(data) < period:
            return np.full_like(data, np.nan)
        result = np.full_like(data, np.nan)
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current Price) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw_raw = smma(close, 13)
    teeth_raw = smma(close, 8)
    lips_raw = smma(close, 5)
    
    # Shift forward: Jaw by 8, Teeth by 5, Lips by 3
    jaw = np.full_like(jaw_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    if len(jaw) > 8:
        jaw[8:] = jaw_raw[:-8]
    if len(teeth) > 5:
        teeth[5:] = teeth_raw[:-5]
    if len(lips) > 3:
        lips[3:] = lips_raw[:-3]
    
    # Calculate ADX (Average Directional Index)
    def calculate_adx(high, low, close, period=14):
        """Calculate ADX, +DI, -DI"""
        if len(high) < period + 1:
            return np.full_like(high, np.nan), np.full_like(high, np.nan), np.full_like(high, np.nan)
        
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # First TR is undefined
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed TR, +DM, -DM (using Wilder's smoothing)
        def wilders_smoothing(data, period):
            if len(data) < period:
                return np.full_like(data, np.nan)
            result = np.full_like(data, np.nan)
            # First value is SMA
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values
            for i in range(period, len(data)):
                if np.isnan(result[i-1]):
                    result[i] = np.nan
                else:
                    result[i] = (result[i-1] * (period-1) + data[i]) / period
            return result
        
        tr_smoothed = wilders_smoothing(tr, period)
        plus_dm_smoothed = wilders_smoothing(plus_dm, period)
        minus_dm_smoothed = wilders_smoothing(minus_dm, period)
        
        # +DI and -DI
        plus_di = np.where(tr_smoothed != 0, (plus_dm_smoothed / tr_smoothed) * 100, 0)
        minus_di = np.where(tr_smoothed != 0, (minus_dm_smoothed / tr_smoothed) * 100, 0)
        
        # DX and ADX
        dx = np.where((plus_di + minus_di) != 0, 
                      np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
        adx = wilders_smoothing(dx, period)
        
        return adx, plus_di, minus_di
    
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        adx_val = adx[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        curr_close = close[i]
        
        # Alligator alignment conditions
        bullish_alignment = jaw_val < teeth_val < lips_val
        bearish_alignment = jaw_val > teeth_val > lips_val
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when: ADX > 25 + bullish alignment + close > lips + volume confirmation
            if adx_val > 25 and bullish_alignment and curr_close > lips_val and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when: ADX > 25 + bearish alignment + close < lips + volume confirmation
            elif adx_val > 25 and bearish_alignment and curr_close < lips_val and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when trend weakens or alignment breaks
            if adx_val < 20 or not (jaw_val < teeth_val < lips_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when trend weakens or alignment breaks
            if adx_val < 20 or not (jaw_val > teeth_val > lips_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
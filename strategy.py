#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot levels (S1, R1) from 1d + volume spike + ADX trend filter.
# Camarilla levels act as intraday support/resistance. Price touching S1/R1 with rejection
# (close back inside range) offers mean reversion entries. In strong trends (ADX>25),
# breakouts of S1/R1 with volume confirmation capture momentum. Combined, this adapts
# to ranging and trending markets. Volume spike (>2x 20-period average) filters low-
# conviction moves. Designed for low trade frequency (~20-30/year) to minimize fee decay.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data for Camarilla calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.1 * (high - low)
    # R2 = close + 0.6 * (high - low)
    # R1 = close + 0.318 * (high - low)
    # S1 = close - 0.318 * (high - low)
    # S2 = close - 0.6 * (high - low)
    # S3 = close - 1.1 * (high - low)
    # S4 = close - 1.5 * (high - low)
    # We use S1 and R1 for entries
    range_1d = high_1d - low_1d
    r1 = close_1d + 0.318 * range_1d
    s1 = close_1d - 0.318 * range_1d
    
    # Calculate ADX on 1d for trend strength filter
    # ADX requires +DI and -DI calculation
    # +DM = max(high - previous high, 0) if high - previous high > previous low - low else 0
    # -DM = max(previous low - low, 0) if previous low - low > high - previous high else 0
    # TR = max(high - low, high - previous close, previous close - low)
    # +DM smoothed, -DM smoothed, TR smoothed over 14 periods
    # +DI = 100 * smoothed +DM / smoothed TR
    # -DI = 100 * smoothed -DM / smoothed TR
    # DX = 100 * |+DI - -DI| / (+DI + -DI)
    # ADX = smoothed DX over 14 periods
    
    # Calculate +DM and -DM
    high_diff = np.diff(np.concatenate([[high_1d[0]], high_1d]))  # high - previous high
    low_diff = np.diff(np.concatenate([[low_1d[0]], low_1d]))     # low - previous low
    plus_dm = np.where((high_diff > -low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((-low_diff > high_diff) & (-low_diff > 0), -low_diff, 0.0)
    
    # Calculate True Range
    prev_close = np.concatenate([[close_1d[0]], close_1d[:-1]])
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - prev_close)
    tr3 = np.abs(low_1d - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        for i in range(len(data)):
            if np.isnan(result[i-1]) if i > 0 else True:
                result[i] = data[i]
            else:
                result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result
    
    period = 14
    tr_smoothed = wilders_smoothing(tr, period)
    plus_dm_smoothed = wilders_smoothing(plus_dm, period)
    minus_dm_smoothed = wilders_smoothing(minus_dm, period)
    
    # Avoid division by zero
    plus_di = np.where(tr_smoothed != 0, 100 * plus_dm_smoothed / tr_smoothed, 0)
    minus_di = np.where(tr_smoothed != 0, 100 * minus_dm_smoothed / tr_smoothed, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smoothing(dx, period)
    
    # Align 1d indicators to 4h timeframe (waits for 1d bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        adx_val = adx_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_val > 25
        
        if position == 0:
            # Mean reversion in ranging market: price touches S1/R1 and reverses
            # Long: price touches or goes below S1 then closes back above S1
            # Short: price touches or goes above R1 then closes back below R1
            # Use previous close to detect reversal
            if i > 0:
                prev_close = prices['close'].iloc[i-1]
                # Long reversal from S1
                if prev_close <= s1_val and price > s1_val and not strong_trend:
                    signals[i] = 0.25
                    position = 1
                # Short reversal from R1
                elif prev_close >= r1_val and price < r1_val and not strong_trend:
                    signals[i] = -0.25
                    position = -1
            # Breakout in trending market: price breaks S1/R1 with volume
            # Long: price breaks above R1 with volume
            # Short: price breaks below S1 with volume
            elif strong_trend and vol_spike:
                if price > r1_val:
                    signals[i] = 0.25
                    position = 1
                elif price < s1_val:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price reaches opposite level (R1) or trend weakens
                if price >= r1_val or adx_val < 20:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price reaches opposite level (S1) or trend weakens
                if price <= s1_val or adx_val < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_S1R1_1dADX_Volume"
timeframe = "4h"
leverage = 1.0
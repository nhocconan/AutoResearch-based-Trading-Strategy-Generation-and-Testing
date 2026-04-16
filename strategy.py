#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily data for pivot points ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Pivot, R1, S1 levels (standard formula)
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    r1 = pivot + range_hl * 0.382
    s1 = pivot - range_hl * 0.382
    
    # === 4h EMA for trend filter (34-period) ===
    ema_4h = pd.Series(close).ewm(span=34, min_periods=34, adjust=False).mean().values
    
    # Align HTF data to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Volume spike detection (15-period volume MA) ===
    vol_ma = pd.Series(volume).rolling(window=15, min_periods=15).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # === Choppiness regime filter (14-period) ===
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low to avoid roll artifact
    tr[0] = tr1[0]
    
    # Calculate +DI and -DI
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    # Set first values to 0
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    atr = np.zeros_like(tr)
    plus_di_smooth = np.zeros_like(tr)
    minus_di_smooth = np.zeros_like(tr)
    
    # Initialize first values
    atr[0] = tr[0]
    plus_di_smooth[0] = plus_dm[0]
    minus_di_smooth[0] = minus_dm[0]
    
    # Wilder smoothing
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
        plus_di_smooth[i] = (plus_di_smooth[i-1] * 13 + plus_dm[i]) / 14
        minus_di_smooth[i] = (minus_di_smooth[i-1] * 13 + minus_dm[i]) / 14
    
    # Calculate DI values
    plus_di = np.where(atr != 0, plus_di_smooth / atr * 100, 0)
    minus_di = np.where(atr != 0, minus_di_smooth / atr * 100, 0)
    
    # Calculate DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = np.zeros_like(dx)
    adx[0] = dx[0]
    for i in range(1, len(dx)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Chop = log(sum(TR,14)/ (max(high,14)-min(low,14))) / log(14) * 100
    # Using rolling window for efficiency
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where((max_high - min_low) != 0, 
                    np.log10(tr_sum / (max_high - min_low)) / np.log10(14) * 100, 50)
    chop[np.isnan(chop)] = 50  # Default to neutral chop
    
    # Chop regime: >61.8 = ranging (mean revert), <38.2 = trending (trend follow)
    chopping = chop > 61.8  # True when chopping/ranging
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or
            np.isnan(ema_4h[i]) or np.isnan(volume_spike[i]) or
            np.isnan(chopping[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r1_level = r1_4h[i]
        s1_level = s1_4h[i]
        ema_val = ema_4h[i]
        vol_spike = volume_spike[i]
        is_chopping = chopping[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price drops below S1 OR chop regime ends
            if price < s1_level or not is_chopping:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price rises above R1 OR chop regime ends
            if price > r1_level or not is_chopping:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # In chopping regime, mean revert at pivot levels
            if is_chopping:
                # LONG: Price touches S1 with volume spike
                if abs(price - s1_level) < (r1_level - s1_level) * 0.02 and vol_spike:
                    signals[i] = 0.25
                    position = 1
                    continue
                # SHORT: Price touches R1 with volume spike
                elif abs(price - r1_level) < (r1_level - s1_level) * 0.02 and vol_spike:
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Pivot_R1_S1_Chop_MeanReversion_Volume"
timeframe = "4h"
leverage = 1.0
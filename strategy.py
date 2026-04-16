#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R mean reversion with 1d volume spike filter and 1w ADX regime filter.
# Long when Williams %R(14) crosses above -80 (oversold bounce) AND volume > 1.5x 20-period average AND 1w ADX > 20.
# Short when Williams %R(14) crosses below -20 (overbought rejection) AND volume > 1.5x 20-period average AND 1w ADX > 20.
# Exit when Williams %R returns to -50 (mean reversion midpoint) or opposite signal occurs.
# Uses discrete position size 0.25. Williams %R identifies exhaustion points, volume confirms participation,
# and 1w ADX ensures we trade in sufficient trending/volatile conditions. Target: 40-80 total trades over 4 years (10-20/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: Williams %R(14) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close_1d) / (highest_high - lowest_low)) * -100,
        -50  # neutral when range is zero
    )
    
    # Align Williams %R to 1d timeframe (no shift as it's based on completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Get 1w data once before loop for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === 1w Indicators: ADX(14) for regime filter ===
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where(np.isnan(dx), 0, dx)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 1d timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        wr = williams_r_aligned[i]
        adx_val = adx_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Calculate 20-period volume average
        if i >= 20:
            vol_ma_20 = np.mean(volume[max(0, i-19):i+1])
        else:
            vol_ma_20 = 0.0
        
        # Volume filter: volume > 1.5x 20-period average
        vol_filter = vol > 1.5 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Regime filter: 1w ADX > 20 (sufficient trend/volatility)
        regime_filter = adx_val > 20
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Williams %R returns to -50 (mean reversion) or crosses below -80
            if wr <= -50:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Williams %R returns to -50 (mean reversion) or crosses above -20
            if wr >= -50:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Williams %R cross above -80 (oversold bounce) with volume and regime confirmation
            if i > warmup:
                wr_prev = williams_r_aligned[i-1]
                if wr > -80 and wr_prev <= -80 and vol_filter and regime_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
            
            # Williams %R cross below -20 (overbought rejection) with volume and regime confirmation
            elif i > warmup:
                wr_prev = williams_r_aligned[i-1]
                if wr < -20 and wr_prev >= -20 and vol_filter and regime_filter:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_WilliamsR_OversoldBounce_1dVolumeSpike_1wADXRegime_V1"
timeframe = "1d"
leverage = 1.0
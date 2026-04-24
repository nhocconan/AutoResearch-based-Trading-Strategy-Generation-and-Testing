#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band Width regime + 1d Camarilla pivot breakout with volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for Camarilla pivot levels (structure) and Bollinger Band Width percentile (regime filter).
- Logic: In low volatility regime (BBW < 30th percentile), trade Camarilla breakouts:
         Long when price breaks above R4 with volume > 1.5 * 20-period average.
         Short when price breaks below S4 with volume > 1.5 * 20-period average.
         In high volatility regime (BBW > 70th percentile), fade at R3/S3:
         Long when price crosses below R3 with volume confirmation.
         Short when price crosses above S3 with volume confirmation.
- Exit: Opposite Camarilla level cross OR BBW regime flip.
- Signal size: 0.25 discrete to minimize fee drag.
- Why it works: BBW regime identifies chop vs trend environments. Camarilla levels provide
   institutional support/resistance. Volume confirmation ensures participation. Works in
   both bull (breakouts in low vol) and bear (fades in high vol) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and Band Width."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean()
    std = pd.Series(close).rolling(window=period, min_periods=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    bbw = (upper - lower) / (sma + 1e-10)  # Band Width as ratio
    return upper.values, lower.values, bbw.values

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels."""
    pivot = (high + low + close) / 3.0
    range_ = high - low
    r3 = pivot + (range_ * 1.1 / 4.0)
    r4 = pivot + (range_ * 1.1 / 2.0)
    s3 = pivot - (range_ * 1.1 / 4.0)
    s4 = pivot - (range_ * 1.1 / 2.0)
    return r3, r4, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need sufficient data
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data for Camarilla pivots and BBW regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    camarilla_data = []
    for i in range(len(df_1d)):
        r3, r4, s3, s4 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        camarilla_data.append([r3, r4, s3, s4])
    
    camarilla_array = np.array(camarilla_data)
    r3_1d = camarilla_array[:, 0]
    r4_1d = camarilla_array[:, 1]
    s3_1d = camarilla_array[:, 2]
    s4_1d = camarilla_array[:, 3]
    
    # Align Camarilla levels to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate 1d Bollinger Band Width for regime filter
    _, _, bbw_1d = bollinger_bands(close_1d, 20, 2.0)
    
    # Calculate BBW percentile rank (using 50-period lookback)
    bbw_percentile = np.full_like(bbw_1d, np.nan)
    for i in range(50, len(bbw_1d)):
        if not np.isnan(bbw_1d[i-50:i+1]).any():
            bbw_percentile[i] = (np.sum(bbw_1d[i-50:i] <= bbw_1d[i]) / 50.0) * 100.0
    
    # Align BBW percentile to 6h timeframe
    bbw_percentile_aligned = align_htf_to_ltf(prices, df_1d, bbw_percentile)
    
    # Calculate 1d volume average for confirmation
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for BBW percentile
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(bbw_percentile_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_bbw_percentile = bbw_percentile_aligned[i]
        
        # Get current Camarilla levels
        r3 = r3_1d_aligned[i]
        r4 = r4_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        s4 = s4_1d_aligned[i]
        
        # Get volume threshold (1.5 * 20-period average)
        vol_idx = min(i, len(vol_ma_20)-1) if len(vol_ma_20) > 0 else 0
        vol_threshold = 1.5 * vol_ma_20[vol_idx] if vol_ma_20[vol_idx] > 0 else np.inf
        volume_confirmed = curr_volume > vol_threshold
        
        # Exit conditions
        if position != 0:
            exit_signal = False
            
            if position == 1:  # Long position
                # Exit based on regime
                if curr_bbw_percentile < 30:  # Low vol regime - breakout
                    if curr_close < r4:  # Price falls below breakout level
                        exit_signal = True
                else:  # High vol regime - fade
                    if curr_close > r3:  # Price moves back above fade level
                        exit_signal = True
            elif position == -1:  # Short position
                # Exit based on regime
                if curr_bbw_percentile < 30:  # Low vol regime - breakout
                    if curr_close > s4:  # Price rises above breakout level
                        exit_signal = True
                else:  # High vol regime - fade
                    if curr_close < s3:  # Price moves back below fade level
                        exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions
        if position == 0:
            # Low volatility regime (BBW < 30th percentile) - trade breakouts
            if curr_bbw_percentile < 30:
                # Long breakout: price above R4 with volume confirmation
                if curr_close > r4 and volume_confirmed:
                    signals[i] = 0.25
                    position = 1
                # Short breakout: price below S4 with volume confirmation
                elif curr_close < s4 and volume_confirmed:
                    signals[i] = -0.25
                    position = -1
            # High volatility regime (BBW > 70th percentile) - fade at extremes
            elif curr_bbw_percentile > 70:
                # Long fade: price below R3 with volume confirmation (mean reversion down)
                if curr_close < r3 and volume_confirmed:
                    signals[i] = 0.25
                    position = 1
                # Short fade: price above S3 with volume confirmation (mean reversion up)
                elif curr_close > s3 and volume_confirmed:
                    signals[i] = -0.25
                    position = -1
        
        # Maintain position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "6h_BBWRegime_CamarillaBreakoutFade_1dVOLUME_v1"
timeframe = "6h"
leverage = 1.0
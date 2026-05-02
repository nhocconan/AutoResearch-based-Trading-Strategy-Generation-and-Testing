#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses 1d timeframe for trend direction (EMA34) and volume confirmation (2.0x 24-period average)
# 6h only for precise entry timing via Camarilla R3/S3 breakouts from previous bar
# Volume confirmation ensures institutional participation and reduces false breakouts
# Target: 50-150 total trades over 4 years = 12-37/year for 6h timeframe
# Works in bull markets via trend-aligned breakouts, in bear via EMA filter avoiding counter-trend trades
# Designed for low trade frequency to minimize fee drag (critical for lower timeframes)

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
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
    
    # Pre-compute session hours (08-20 UTC) - index is DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for EMA trend filter and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d volume confirmation (2.0x 24-period average)
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=24, min_periods=24).mean().values
    volume_confirm_1d = volume_1d > (vol_ma_1d * 2.0)
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d.astype(float))
    
    # Calculate Camarilla pivot levels from previous 6h bar
    # Typical Price = (H + L + C)/3
    typical_price = (high + low + close) / 3.0
    # Camarilla levels based on previous bar's range
    rng = high - low
    camarilla_h4 = typical_price + 1.1 * rng / 2.0  # R3
    camarilla_l4 = typical_price - 1.1 * rng / 2.0  # S3
    camarilla_h2 = typical_price + 1.1 * rng / 6.0  # R1
    camarilla_l2 = typical_price - 1.1 * rng / 6.0  # S1
    
    # Shift to align with bar close (use previous bar's levels)
    camarilla_h4 = np.roll(camarilla_h4, 1)
    camarilla_l4 = np.roll(camarilla_l4, 1)
    camarilla_h2 = np.roll(camarilla_h2, 1)
    camarilla_l2 = np.roll(camarilla_l2, 1)
    camarilla_h4[0] = np.nan
    camarilla_l4[0] = np.nan
    camarilla_h2[0] = np.nan
    camarilla_l2[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_confirm_1d_aligned[i]) or 
            np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or
            np.isnan(camarilla_h2[i]) or np.isnan(camarilla_l2[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Camarilla H4 (R3) + price > 1d EMA34 + volume confirm
            if close[i] > camarilla_h4[i] and close[i] > ema_34_1d_aligned[i] and volume_confirm_1d_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla L4 (S3) + price < 1d EMA34 + volume confirm
            elif close[i] < camarilla_l4[i] and close[i] < ema_34_1d_aligned[i] and volume_confirm_1d_aligned[i] > 0.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Camarilla H2 (R1) or trend reversal
            if close[i] < camarilla_h2[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Camarilla L2 (S1) or trend reversal
            if close[i] > camarilla_l2[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
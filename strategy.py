#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses 4h timeframe for signal generation and 1d for trend filter (price > EMA34 = bullish, < EMA34 = bearish)
# Volume confirmation (2.0x 20-period average) ensures institutional participation
# Chop regime filter (1d Chop < 61.8) avoids ranging markets where breakouts fail
# Target: 80-180 total trades over 4 years = 20-45/year for 4h timeframe
# Works in bull markets via trend-aligned breakouts, in bear via chop regime filter avoiding false signals
# Discrete position sizing (0.25) minimizes fee churn while maintaining adequate exposure

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
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
    
    # Load 1d data ONCE before loop for EMA trend filter and Chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Chopiness Index (14) - trending when < 38.2, ranging when > 61.8
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR14
    atr1 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Chop = 100 * log15(sum(ATR14)/ (max(high)-min(low)) over 14 periods)
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log15(atr1 * 14 / (max_high - min_low))
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation (2.0x 20-period average on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    # Calculate Camarilla pivot levels from previous 4h bar
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
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or
            np.isnan(camarilla_h2[i]) or np.isnan(camarilla_l2[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when Chop < 61.8 (not strongly ranging)
        if chop_aligned[i] > 61.8:
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Camarilla H4 (R3) + price > 1d EMA34 + volume confirm
            if close[i] > camarilla_h4[i] and close[i] > ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla L4 (S3) + price < 1d EMA34 + volume confirm
            elif close[i] < camarilla_l4[i] and close[i] < ema_34_1d_aligned[i] and volume_confirm[i]:
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
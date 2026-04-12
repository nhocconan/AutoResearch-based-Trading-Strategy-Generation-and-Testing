#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_Pivot_Bounce_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].iloc[-2] if len(df_1d) >= 2 else df_1d['high'].iloc[-1]
    prev_low = df_1d['low'].iloc[-2] if len(df_1d) >= 2 else df_1d['low'].iloc[-1]
    prev_close = df_1d['close'].iloc[-2] if len(df_1d) >= 2 else df_1d['close'].iloc[-1]
    
    # Calculate Camarilla pivot levels (daily)
    range_val = prev_high - prev_low
    if range_val <= 0:
        return np.zeros(n)
    
    # Camarilla levels (based on previous day)
    camarilla_h5 = prev_close + range_val * 1.1 / 2
    camarilla_h4 = prev_close + range_val * 1.1
    camarilla_h3 = prev_close + range_val * 1.1 * 1.5
    camarilla_l3 = prev_close - range_val * 1.1 * 1.5
    camarilla_l4 = prev_close - range_val * 1.1
    camarilla_l5 = prev_close - range_val * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    h5_array = np.full(len(df_1d), camarilla_h5)
    h4_array = np.full(len(df_1d), camarilla_h4)
    h3_array = np.full(len(df_1d), camarilla_h3)
    l3_array = np.full(len(df_1d), camarilla_l3)
    l4_array = np.full(len(df_1d), camarilla_l4)
    l5_array = np.full(len(df_1d), camarilla_l5)
    
    h5_aligned = align_htf_to_ltf(prices, df_1d, h5_array)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4_array)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3_array)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3_array)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4_array)
    l5_aligned = align_htf_to_ltf(prices, df_1d, l5_array)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = volume_series / vol_ma
    vol_ratio = vol_ratio.fillna(1.0).values
    
    # Choppy market filter: use intraday range vs ATR
    # Calculate ATR(14) on 12h data for volatility normalization
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Intraday range as percentage of ATR
    intraday_range = (high - low) / atr
    intraday_range = np.where(atr == 0, 1.0, intraday_range)
    chop_threshold = 2.0  # High intraday range relative to ATR = trending
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(h5_aligned[i]) or np.isnan(h4_aligned[i]) or np.isnan(h3_aligned[i]) or
            np.isnan(l3_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(l5_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(intraday_range[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Only trade in less choppy (more trending) markets
        is_trending = intraday_range[i] > chop_threshold
        
        # Mean reversion at extreme Camarilla levels with volume confirmation
        # Long near S3/S4 when oversold and volume confirms buying interest
        long_setup = ((close[i] <= l3_aligned[i] * 1.002) or (close[i] <= l4_aligned[i] * 1.002)) and \
                     vol_ratio[i] > 1.3 and is_trending
        
        # Short near R3/R4 when overbought and volume confirms selling pressure
        short_setup = ((close[i] >= h3_aligned[i] * 0.998) or (close[i] >= h4_aligned[i] * 0.998)) and \
                      vol_ratio[i] > 1.3 and is_trending
        
        # Exit when price returns to central zone (H3-L3)
        long_exit = close[i] >= h3_aligned[i] * 0.995
        short_exit = close[i] <= l3_aligned[i] * 1.005
        
        # Signal logic
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
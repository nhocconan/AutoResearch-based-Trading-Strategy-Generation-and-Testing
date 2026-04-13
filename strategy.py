#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h primary with 1d HTF - Camarilla H3/L3 mean reversion with volume confirmation
    # In ranging markets, price tends to revert to the mean after touching H3/L3 levels
    # Volume confirmation filters false breakouts; only trade when volume supports the move
    # Works in both bull and bear markets by fading extremes in a choppy regime
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for HTF Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 4h data for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    volume_4h = df_4h['volume'].values if 'volume' in df_4h.columns else np.ones(len(df_4h))
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    
    # Camarilla H3/L3 levels (mean reversion targets)
    camarilla_h3 = prev_close_1d + 1.125 * (prev_high_1d - prev_low_1d)
    camarilla_l3 = prev_close_1d - 1.125 * (prev_high_1d - prev_low_1d)
    
    # Calculate 1d EMA20 for trend filter (avoid strong trends)
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h close for price reference
    df_4h_close = get_htf_data(prices, '4h')['close'].values
    
    # Align all HTF indicators to 4h primary timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_4h, vol_avg_20)
    close_4h_aligned = align_htf_to_ltf(prices, df_4h, df_4h_close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema20_1d_aligned[i]) or
            np.isnan(vol_avg_20_aligned[i]) or
            np.isnan(close_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume_4h[i] > 1.5 * vol_avg_20_aligned[i]
        
        # Mean reversion conditions at Camarilla H3/L3 levels
        touch_h3 = close_4h_aligned[i] >= camarilla_h3_aligned[i]
        touch_l3 = close_4h_aligned[i] <= camarilla_l3_aligned[i]
        
        # Trend filter: only trade when price is near EMA20 (avoid strong trends)
        near_ema = abs(close_4h_aligned[i] - ema20_1d_aligned[i]) / ema20_1d_aligned[i] < 0.02
        
        # Entry conditions: fade the extreme when volume confirms
        enter_long = touch_l3 and volume_confirmed and near_ema
        enter_short = touch_h3 and volume_confirmed and near_ema
        
        # Exit conditions: price returns to the opposite H3/L3 level or crosses EMA20
        exit_long = position == 1 and (close_4h_aligned[i] <= camarilla_h3_aligned[i] or 
                                       close_4h_aligned[i] >= ema20_1d_aligned[i])
        exit_short = position == -1 and (close_4h_aligned[i] >= camarilla_l3_aligned[i] or 
                                         close_4h_aligned[i] <= ema20_1d_aligned[i])
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_h3l3_meanrev_volume_v2"
timeframe = "4h"
leverage = 1.0
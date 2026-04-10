#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and ADX trend filter
# - Long when price breaks above Camarilla H3 level (1d) + ADX(14) > 25 (trending) + 1d volume > 1.5x 20-period volume SMA
# - Short when price breaks below Camarilla L3 level (1d) + same ADX and volume conditions
# - Exit: price returns to Camarilla pivot point (mean of H3/L3)
# - Position sizing: 0.25 discrete level
# - Camarilla levels provide intraday support/resistance with clear breakout zones
# - ADX filter ensures we only trade in trending markets, avoiding chop
# - Volume confirmation ensures breakout strength
# - Target: 20-40 trades/year to minimize fee drag while capturing strong moves

name = "4h_1d_camarilla_adx_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla: H4 = C + 1.5*(H-L), H3 = C + 1.0*(H-L), L3 = C - 1.0*(H-L), L4 = C - 1.5*(H-L)
    # Pivot = (H+L+C)/3
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    camarilla_h3 = prev_close + 1.0 * (prev_high - prev_low)
    camarilla_l3 = prev_close - 1.0 * (prev_high - prev_low)
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3.0
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Calculate 4h ADX for trend filter
    # True Range
    tr1 = np.maximum(high - low, 
                     np.maximum(np.abs(high - np.roll(close, 1)), 
                                np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr_period = 14
    atr = pd.Series(tr1).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=atr_period, min_periods=atr_period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=atr_period, min_periods=atr_period).mean().values / atr
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Calculate 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(adx[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume for volume spike confirmation
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        
        # Volume confirmation: current 1d volume > 1.5x 20-period SMA (strong volume spike)
        vol_confirm = vol_1d_current[i] > 1.5 * volume_sma_20_1d_aligned[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx[i] > 25
        
        # Camarilla breakout signals
        long_entry = (close[i] > camarilla_h3_aligned[i]) and trending and vol_confirm
        short_entry = (close[i] < camarilla_l3_aligned[i]) and trending and vol_confirm
        exit_long = close[i] < camarilla_pivot_aligned[i]  # Exit long when price crosses below pivot
        exit_short = close[i] > camarilla_pivot_aligned[i]  # Exit short when price crosses above pivot
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals
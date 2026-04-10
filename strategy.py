#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 12h volume spike and chop regime filter
# - Long when price breaks above Camarilla H3 level AND 4h volume > 2x 20-period volume SMA AND chop < 61.8 (trending)
# - Short when price breaks below Camarilla L3 level AND 4h volume > 2x 20-period volume SMA AND chop < 61.8 (trending)
# - Exit: opposite Camarilla breakout or volume drops below average or chop > 61.8 (range)
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 20-50 trades/year on 4h timeframe to stay within fee drag limits

name = "4h_12h_camarilla_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 4h Camarilla pivot levels (based on previous day)
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # We'll use H3 and L3 for breakouts
    # For each bar, we need the previous day's OHLC
    # Since we're on 4h timeframe, we'll use daily data shifted appropriately
    
    # Get daily OHLC aligned to 4h bars
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Align daily data to 4h bars (each daily bar covers 6x 4h bars)
    open_1d_aligned = align_htf_to_ltf(prices, df_1d, open_1d)
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Calculate Camarilla levels for each 4h bar using previous day's OHLC
    # H3 = close + 1.1*(high - low)/4
    # L3 = close - 1.1*(high - low)/4
    # We need to use the previous day's values, so shift by 1
    prev_close = close_1d_aligned
    prev_high = high_1d_aligned
    prev_low = low_1d_aligned
    
    # Shift to get previous day's values (avoiding look-ahead)
    prev_close_shifted = np.roll(prev_close, 6)  # 6*4h = 24h = 1 day
    prev_high_shifted = np.roll(prev_high, 6)
    prev_low_shifted = np.roll(prev_low, 6)
    
    # Set first 6 values to NaN (no previous day data)
    prev_close_shifted[:6] = np.nan
    prev_high_shifted[:6] = np.nan
    prev_low_shifted[:6] = np.nan
    
    camarilla_h3 = prev_close_shifted + 1.1 * (prev_high_shifted - prev_low_shifted) / 4
    camarilla_l3 = prev_close_shifted - 1.1 * (prev_high_shifted - prev_low_shifted) / 4
    
    # Calculate 12h EMA20 for trend filter (optional, can remove if too restrictive)
    close_12h = df_12h['close'].values
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Calculate 4h volume SMA for regime filter
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR over n periods) / (log10(n) * (max(high,n) - min(low,n))))
    # Simplified version: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    atr_period = 14
    chop_period = 14
    
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Calculate ATR
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Calculate sum of ATR over chop_period
    atr_sum = pd.Series(atr).rolling(window=chop_period, min_periods=chop_period).sum().values
    
    # Calculate max(high) and min(low) over chop_period
    max_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    min_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    
    # Calculate Choppiness Index
    # Avoid division by zero and log of zero
    denominator = np.log10(chop_period) * (max_high - min_low)
    chop = np.where(denominator != 0, 100 * np.log10(atr_sum) / denominator, 50)
    # Set to 50 (neutral) when invalid
    chop = np.where((max_high - min_low) == 0, 50, chop)
    
    for i in range(50, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(volume_sma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 4h volume > 2x 20-period volume SMA
        vol_confirm = volume[i] > 2.0 * volume_sma_20[i]
        
        # Chop regime filter: CHOP < 61.8 = trending (favor breakouts)
        chop_filter = chop[i] < 61.8
        
        # Camarilla breakout signals
        breakout_up = close[i] > camarilla_h3[i]  # Break above H3 level
        breakout_down = close[i] < camarilla_l3[i]  # Break below L3 level
        
        # Exit conditions: opposite breakout or loss of volume confirmation or chop > 61.8 (range)
        exit_long = breakout_down or not vol_confirm or chop[i] >= 61.8
        exit_short = breakout_up or not vol_confirm or chop[i] >= 61.8
        
        if position == 0:  # Flat - look for entry
            if breakout_up and vol_confirm and chop_filter:
                position = 1
                signals[i] = 0.25
            elif breakout_down and vol_confirm and chop_filter:
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
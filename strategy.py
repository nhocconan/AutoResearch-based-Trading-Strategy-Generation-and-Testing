#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 12h volume confirmation and ADX trend filter
# - Long when price breaks above Camarilla H3 level AND 12h volume > 1.2x 20-period 12h volume SMA AND ADX(14) > 20
# - Short when price breaks below Camarilla L3 level AND 12h volume > 1.2x 20-period 12h volume SMA AND ADX(14) > 20
# - Exit: price retrace to Camarilla H4/L4 levels or opposite breakout with volume confirmation
# - Position sizing: 0.25 discrete level
# - Uses 12h HTF for volume and trend alignment to reduce false breakouts in ranging markets
# - Camarilla levels derived from 1d OHLC for institutional reference points
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)

name = "4h_12h_camarilla_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 12h HTF data for volume and trend filters
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h volume SMA for confirmation
    volume_12h = df_12h['volume'].values
    volume_sma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_sma_20_12h)
    
    # 12h ADX for trend strength filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate True Range for 12h
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    tr_12h[0] = tr1_12h[0]  # First period TR
    
    # Calculate +DM and -DM for 12h
    up_move_12h = high_12h - np.roll(high_12h, 1)
    down_move_12h = np.roll(low_12h, 1) - low_12h
    up_move_12h[0] = 0
    down_move_12h[0] = 0
    
    plus_dm_12h = np.where((up_move_12h > down_move_12h) & (up_move_12h > 0), up_move_12h, 0)
    minus_dm_12h = np.where((down_move_12h > up_move_12h) & (down_move_12h > 0), down_move_12h, 0)
    
    # Smooth TR, +DM, -DM with Wilder's smoothing (using EMA as approximation)
    atr_12h = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di_12h = 100 * pd.Series(plus_dm_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_12h
    minus_di_12h = 100 * pd.Series(minus_dm_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_12h
    
    # Calculate ADX
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    dx_12h[np.isnan(dx_12h) | np.isinf(dx_12h)] = 0
    adx_12h = pd.Series(dx_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate 1d HTF data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d OHLC
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # Based on previous day's range: H/L = (High - Low)
    # H4 = Close + 1.5 * (High - Low) * 1.1/2
    # H3 = Close + 1.25 * (High - Low) * 1.1/2
    # H2 = Close + 1.166 * (High - Low) * 1.1/2
    # H1 = Close + 1.083 * (High - Low) * 1.1/2
    # L1 = Close - 1.083 * (High - Low) * 1.1/2
    # L2 = Close - 1.166 * (High - Low) * 1.1/2
    # L3 = Close - 1.25 * (High - Low) * 1.1/2
    # L4 = Close - 1.5 * (High - Low) * 1.1/2
    
    prev_high_1d = df_1d['high'].values
    prev_low_1d = df_1d['low'].values
    prev_close_1d = df_1d['close'].values
    
    # Calculate Camarilla H3 and L3 levels (primary breakout levels)
    camarilla_h3_1d = prev_close_1d + 1.25 * (prev_high_1d - prev_low_1d) * 1.1 / 2
    camarilla_l3_1d = prev_close_1d - 1.25 * (prev_high_1d - prev_low_1d) * 1.1 / 2
    
    # Calculate Camarilla H4 and L4 levels (exit levels)
    camarilla_h4_1d = prev_close_1d + 1.5 * (prev_high_1d - prev_low_1d) * 1.1 / 2
    camarilla_l4_1d = prev_close_1d - 1.5 * (prev_high_1d - prev_low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d)
    
    # Calculate 4h volume SMA for confirmation (secondary)
    volume_sma_20_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(20, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(volume_12h_aligned[i]) or np.isnan(adx_12h_aligned[i]) or
            np.isnan(volume_sma_20_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 1.2x 20-period 12h volume SMA OR 4h volume > 1.5x 20-period 4h volume SMA
        vol_confirm_12h = volume_12h_aligned[i] > 1.2
        vol_confirm_4h = volume[i] > 1.5 * volume_sma_20_4h[i]
        vol_confirm = vol_confirm_12h or vol_confirm_4h
        
        # Trend filter: ADX(14) > 20 on 12h
        trend_filter = adx_12h_aligned[i] > 20
        
        # Camarilla breakout signals
        breakout_up = close[i] > camarilla_h3_aligned[i]  # Break above H3
        breakout_down = close[i] < camarilla_l3_aligned[i]  # Break below L3
        
        if position == 0:  # Flat - look for entry
            if breakout_up and vol_confirm and trend_filter:
                position = 1
                signals[i] = 0.25
            elif breakout_down and vol_confirm and trend_filter:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            # Exit on retrace to H4 or opposite breakout with volume
            exit_condition = (close[i] < camarilla_h4_aligned[i]) or \
                           (breakout_down and vol_confirm)
            if exit_condition:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            # Exit on retrace to L4 or opposite breakout with volume
            exit_condition = (close[i] > camarilla_l4_aligned[i]) or \
                           (breakout_up and vol_confirm)
            if exit_condition:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike and 1d trend filter (close > EMA50)
# - Long when price breaks above 4h Camarilla R3 level AND 1d volume > 1.5x 20-period volume SMA AND 1d close > 1d EMA50
# - Short when price breaks below 4h Camarilla S3 level AND 1d volume > 1.5x 20-period volume SMA AND 1d close < 1d EMA50
# - Exit: price retreats to Camarilla pivot point (PP) or volume drops below 1.2x SMA
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 20-50 trades/year on 4h timeframe to stay within fee drag limits
# - Uses Camarilla levels from 1d timeframe for structure, 4h for execution timing

name = "4h_1d_camarilla_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Calculate 1d Camarilla pivot levels
    # Camarilla formula: PP = (H + L + C) / 3
    # R3 = PP + (H - L) * 1.1/4
    # S3 = PP - (H - L) * 1.1/4
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    camarilla_range = high_1d - low_1d
    camarilla_r3 = camarilla_pp + camarilla_range * 1.1 / 4.0
    camarilla_s3 = camarilla_pp - camarilla_range * 1.1 / 4.0
    
    # Align 1d Camarilla levels to 4h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate 4h volume SMA for confirmation
    volume_sma_20_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(50, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_sma_20_1d_aligned[i]) or
            np.isnan(volume_sma_20_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 4h volume > 1.2x 20-period volume SMA AND 1d volume > 1.5x 20-period volume SMA
        vol_confirm_4h = volume[i] > 1.2 * volume_sma_20_4h[i]
        vol_confirm_1d = volume_1d[i // 16] > 1.5 * volume_sma_20_1d_aligned[i] if i // 16 < len(volume_1d) else False
        vol_confirm = vol_confirm_4h and vol_confirm_1d
        
        # Trend filter: 1d close vs 1d EMA50
        trend_bullish = close_1d[i // 16] > ema_50_1d_aligned[i] if i // 16 < len(close_1d) else False
        trend_bearish = close_1d[i // 16] < ema_50_1d_aligned[i] if i // 16 < len(close_1d) else False
        
        # Camarilla breakout signals
        breakout_up = close[i] > camarilla_r3_aligned[i-1]  # Break above previous R3
        breakout_down = close[i] < camarilla_s3_aligned[i-1]  # Break below previous S3
        
        # Exit conditions: price retreats to pivot point or loss of volume confirmation
        exit_long = close[i] < camarilla_pp_aligned[i] or not vol_confirm
        exit_short = close[i] > camarilla_pp_aligned[i] or not vol_confirm
        
        if position == 0:  # Flat - look for entry
            if breakout_up and trend_bullish and vol_confirm:
                position = 1
                signals[i] = 0.25
            elif breakout_down and trend_bearish and vol_confirm:
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
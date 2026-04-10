#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h volume confirmation and 1d trend filter
# - Long when price breaks above 1h Camarilla R3 level AND 4h volume > 1.3x 20-period volume SMA AND 1d close > 1d EMA50
# - Short when price breaks below 1h Camarilla S3 level AND 4h volume > 1.3x 20-period volume SMA AND 1d close < 1d EMA50
# - Exit: price retreats to Camarilla pivot point (PP) or volume drops below average
# - Position sizing: 0.20 discrete level to minimize fee drag
# - Session filter: 08-20 UTC to avoid low-liquidity hours
# - Target: 60-150 total trades over 4 years = 15-37/year for 1h
# - Uses Camarilla levels from 1d timeframe for structure, 4h for volume confirmation, 1h for execution timing

name = "1h_4h_1d_camarilla_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_4h = get_htf_data(prices, '4h')
    if len(df_1d) < 30 or len(df_4h) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 1d Camarilla pivot levels from 1d OHLC
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
    
    # Align 1d Camarilla levels to 1h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h volume SMA for confirmation
    volume_4h = df_4h['volume'].values
    volume_sma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_sma_20_4h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(50, n):  # Start after warmup for indicators
        # Skip if any required data is invalid or outside session
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_sma_20_4h_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1h volume > 1.3x 20-period volume SMA AND 4h volume > 1.3x 20-period volume SMA
        vol_confirm_1h = volume[i] > 1.3 * pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        vol_confirm_4h = volume_4h[i // 4] > 1.3 * volume_sma_20_4h_aligned[i] if i // 4 < len(volume_4h) else False
        vol_confirm = vol_confirm_1h and vol_confirm_4h
        
        # Trend filter: 1d close vs 1d EMA50
        trend_bullish = close_1d[i // 24] > ema_50_1d_aligned[i] if i // 24 < len(close_1d) else False
        trend_bearish = close_1d[i // 24] < ema_50_1d_aligned[i] if i // 24 < len(close_1d) else False
        
        # Camarilla breakout signals
        breakout_up = close[i] > camarilla_r3_aligned[i-1]  # Break above previous R3
        breakout_down = close[i] < camarilla_s3_aligned[i-1]  # Break below previous S3
        
        # Exit conditions: price retreats to pivot point or loss of volume confirmation
        exit_long = close[i] < camarilla_pp_aligned[i] or not vol_confirm
        exit_short = close[i] > camarilla_pp_aligned[i] or not vol_confirm
        
        if position == 0:  # Flat - look for entry
            if breakout_up and trend_bullish and vol_confirm:
                position = 1
                signals[i] = 0.20
            elif breakout_down and trend_bearish and vol_confirm:
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
    
    return signals
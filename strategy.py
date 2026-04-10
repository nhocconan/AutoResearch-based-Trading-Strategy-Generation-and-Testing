#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla breakout with 4h volume confirmation and 1d trend filter
# - Long when price breaks above 1h Camarilla R4 level AND 4h volume > 1.2x 20-period volume SMA AND 1d close > 1d EMA50
# - Short when price breaks below 1h Camarilla S4 level AND 4h volume > 1.2x 20-period volume SMA AND 1d close < 1d EMA50
# - Exit: price retreats to Camarilla pivot point (PP) or volume drops below average
# - Position sizing: 0.20 discrete level to minimize fee drag
# - Session filter: 08-20 UTC to avoid low-volume periods
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)

name = "1h_4h_1d_camarilla_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 1h Camarilla pivot levels from previous 1h bar (to avoid look-ahead)
    # Camarilla formula: PP = (H + L + C) / 3
    # R4 = PP + (H - L) * 1.1/2
    # S4 = PP - (H - L) * 1.1/2
    camarilla_pp = (high + low + close) / 3.0
    camarilla_range = high - low
    camarilla_r4 = camarilla_pp + camarilla_range * 1.1 / 2.0
    camarilla_s4 = camarilla_pp - camarilla_range * 1.1 / 2.0
    
    # Use previous bar's levels for breakout (completed bar only)
    camarilla_pp_prev = np.roll(camarilla_pp, 1)
    camarilla_r4_prev = np.roll(camarilla_r4, 1)
    camarilla_s4_prev = np.roll(camarilla_s4, 1)
    camarilla_pp_prev[0] = np.nan
    camarilla_r4_prev[0] = np.nan
    camarilla_s4_prev[0] = np.nan
    
    # Calculate 4h volume SMA for confirmation
    volume_4h = df_4h['volume'].values
    volume_sma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_sma_20_4h)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d close for trend comparison
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    for i in range(60, n):  # Start after warmup for indicators
        # Session filter
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r4_prev[i]) or np.isnan(camarilla_s4_prev[i]) or
            np.isnan(volume_sma_20_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(close_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 4h volume > 1.2x 20-period volume SMA
        vol_confirm = volume[i] > 1.2 * volume_sma_20_4h_aligned[i]
        
        # Trend filter: 1d close vs 1d EMA50
        trend_bullish = close_1d_aligned[i] > ema_50_1d_aligned[i]
        trend_bearish = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # Camarilla breakout signals (using previous bar's levels)
        breakout_up = close[i] > camarilla_r4_prev[i]
        breakout_down = close[i] < camarilla_s4_prev[i]
        
        # Exit conditions: price retreats to pivot point or loss of volume confirmation
        exit_long = close[i] < camarilla_pp[i] or not vol_confirm
        exit_short = close[i] > camarilla_pp[i] or not vol_confirm
        
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
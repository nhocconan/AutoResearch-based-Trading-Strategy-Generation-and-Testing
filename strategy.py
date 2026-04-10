#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume spike and 1d trend filter
# - Long when price breaks above 6h Camarilla R4 level AND 1d volume > 1.5x 20-period volume SMA AND 1d close > 1d EMA50
# - Short when price breaks below 6h Camarilla S4 level AND 1d volume > 1.5x 20-period volume SMA AND 1d close < 1d EMA50
# - Exit: price retreats to Camarilla pivot point (PP)
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 12-37 trades/year on 6h timeframe to stay within fee drag limits
# - Uses Camarilla levels from 1d timeframe for structure, 6h for execution timing
# - Volume spike filter reduces false breakouts; EMA50 ensures trend alignment

name = "6h_1d_camarilla_volspike_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 1d Camarilla pivot levels from previous day OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    camarilla_range = high_1d - low_1d
    camarilla_r4 = camarilla_pp + camarilla_range * 1.1 / 2.0
    camarilla_s4 = camarilla_pp - camarilla_range * 1.1 / 2.0
    
    # Align 1d Camarilla levels to 6h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d volume SMA for spike detection
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate 6h volume for confirmation (optional)
    volume_sma_20_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(50, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.5x 20-period volume SMA (spike detection)
        # Get 1d index for current 6h bar (4x 6h bars = 1d bar)
        idx_1d = i // 4
        if idx_1d >= len(volume_1d):
            vol_confirm = False
        else:
            vol_confirm = volume_1d[idx_1d] > 1.5 * volume_sma_20_1d_aligned[i]
        
        # Trend filter: 1d close vs 1d EMA50
        trend_bullish = close_1d[idx_1d] > ema_50_1d_aligned[i] if idx_1d < len(close_1d) else False
        trend_bearish = close_1d[idx_1d] < ema_50_1d_aligned[i] if idx_1d < len(close_1d) else False
        
        # Camarilla breakout signals (using previous bar's levels to avoid look-ahead)
        breakout_up = close[i] > camarilla_r4_aligned[i-1]  # Break above previous R4
        breakout_down = close[i] < camarilla_s4_aligned[i-1]  # Break below previous S4
        
        # Exit conditions: price retreats to pivot point
        exit_long = close[i] < camarilla_pp_aligned[i]
        exit_short = close[i] > camarilla_pp_aligned[i]
        
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
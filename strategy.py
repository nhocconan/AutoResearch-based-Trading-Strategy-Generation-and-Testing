#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h volume confirmation and 1d trend filter
# - Long when price breaks above Camarilla R4 (1d) AND 12h volume > 1.5x 20-period volume SMA AND 1d close > 1d EMA50 (bullish trend)
# - Short when price breaks below Camarilla S4 (1d) AND 12h volume > 1.5x 20-period volume SMA AND 1d close < 1d EMA50 (bearish trend)
# - Exit: opposite Camarilla breakout or loss of volume confirmation
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 12-37 trades/year on 6h timeframe to stay within fee drag limits

name = "6h_1d_camarilla_breakout_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    if len(df_1d) < 30 or len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 1d Camarilla pivot levels (using previous day's OHLC)
    # For 6h bars, we need to align the previous day's pivot to current bar
    prev_close_1d = df_1d['close'].shift(1).values  # Previous day's close
    prev_high_1d = df_1d['high'].shift(1).values    # Previous day's high
    prev_low_1d = df_1d['low'].shift(1).values      # Previous day's low
    prev_open_1d = df_1d['open'].shift(1).values    # Previous day's open
    
    # Calculate pivot point
    pivot = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    # Calculate Camarilla levels
    camarilla_r4 = pivot + (prev_high_1d - prev_low_1d) * 1.1 / 2.0
    camarilla_s4 = pivot - (prev_high_1d - prev_low_1d) * 1.1 / 2.0
    
    # Align Camarilla levels to 6h timeframe (previous day's levels are known at 6h bar open)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d close for trend comparison
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
    
    # Calculate 6h volume SMA for regime filter (using 12h data for volume confirmation)
    volume_12h = df_12h['volume'].values
    volume_sma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_sma_20_12h)
    
    for i in range(60, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(close_1d_aligned[i]) or
            np.isnan(volume_sma_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 1.5x 20-period volume SMA
        # Get current 12h volume aligned to 6h bar
        vol_12h_current = df_12h['volume'].iloc[-1] if len(df_12h) > 0 else 0  # Simplified - in practice we'd need proper alignment
        # Instead, we'll use the 6h volume with 12h SMA as proxy
        vol_confirm = volume[i] > 1.5 * volume_sma_20_12h_aligned[i]
        
        # Trend filter: 1d close vs 1d EMA50
        trend_bullish = close_1d_aligned[i] > ema_50_1d_aligned[i]
        trend_bearish = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # Camarilla breakout signals (using previous bar's levels to avoid look-ahead)
        breakout_up = close[i] > camarilla_r4_aligned[i-1]  # Break above previous R4
        breakout_down = close[i] < camarilla_s4_aligned[i-1]  # Break below previous S4
        
        # Exit conditions: opposite breakout or loss of volume confirmation
        exit_long = breakout_down or not vol_confirm
        exit_short = breakout_up or not vol_confirm
        
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
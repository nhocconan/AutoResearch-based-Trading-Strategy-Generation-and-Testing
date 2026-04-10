#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d trend filter and volume confirmation
# - Long when price breaks above Camarilla R4 AND 1d close > 1d EMA50 (bullish trend)
# - Short when price breaks below Camarilla S4 AND 1d close < 1d EMA50 (bearish trend)
# - Volume confirmation: 6h volume > 1.3x 20-period volume SMA
# - Exit: price retracement to Camarilla pivot point (PP) or opposite breakout
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 12-37 trades/year on 6h timeframe to stay within fee drag limits

name = "6h_1d_camarilla_breakout_trend_volume_v1"
timeframe = "6h"
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
    
    # Calculate Camarilla pivot levels from previous day
    # PP = (H + L + C) / 3
    # R4 = PP + (H - L) * 1.1/2
    # S4 = PP - (H - L) * 1.1/2
    # We use previous day's OHLC to calculate today's levels
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pp = (prev_high + prev_low + prev_close) / 3.0
    r4 = pp + (prev_high - prev_low) * 1.1 / 2.0
    s4 = pp - (prev_high - prev_low) * 1.1 / 2.0
    pivot_point = pp  # Camarilla pivot point for exit
    
    # Align HTF levels to LTF
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d close for trend comparison
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
    
    # Calculate 6h volume SMA for regime filter
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(50, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(close_1d_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 6h volume > 1.3x 20-period volume SMA
        vol_confirm = volume[i] > 1.3 * volume_sma_20[i]
        
        # Trend filter: 1d close vs 1d EMA50
        trend_bullish = close_1d_aligned[i] > ema_50_1d_aligned[i]
        trend_bearish = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # Camarilla breakout signals
        breakout_up = close[i] > r4_aligned[i]  # Break above R4
        breakout_down = close[i] < s4_aligned[i]  # Break below S4
        
        # Exit conditions: retracement to pivot point or opposite breakout
        exit_long = (close[i] <= pp_aligned[i]) or breakout_down
        exit_short = (close[i] >= pp_aligned[i]) or breakout_up
        
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
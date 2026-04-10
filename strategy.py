#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 12h volume spike and 1d trend filter
# - Enter long when 6h price breaks above upper BB(20,2) AND 12h volume > 2.0x 20-period volume SMA AND 1d close > 1d EMA50
# - Enter short when 6h price breaks below lower BB(20,2) AND 12h volume > 2.0x 20-period volume SMA AND 1d close < 1d EMA50
# - Exit: price returns to BB middle (20-period SMA) or opposite band touch
# - Bollinger squeeze identifies low volatility primed for expansion
# - Volume confirmation ensures breakouts have participation
# - 1d EMA50 filter avoids counter-trend trades in strong trends
# - Target: 12-25 trades/year to minimize fee drag while capturing high-probability breakouts

name = "6h_12h_1d_bb_squeeze_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 12h data ONCE before loop for volume confirmation (MTF rule compliance)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return signals
    
    # Load 1d data ONCE before loop for trend filter (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute Bollinger Bands for 6h data (20-period, 2 std)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    middle_bb = sma_20  # 20-period SMA
    
    # Pre-compute volume SMA for 12h data (20-period)
    volume_12h = df_12h['volume'].values
    volume_sma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_sma_20_12h)
    
    # Pre-compute EMA50 for 1d close (trend filter)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 1d close aligned for trend comparison
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    for i in range(20, n):  # Start after 20-bar warmup for BB
        # Skip if any required data is invalid
        if (np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or np.isnan(middle_bb[i]) or 
            np.isnan(volume_sma_20_12h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(close_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 2.0x 20-period volume SMA (tighter threshold)
        vol_12h_current = df_12h['volume'].values
        vol_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_12h_current)
        vol_confirm = vol_12h_aligned[i] > 2.0 * volume_sma_20_12h_aligned[i]
        
        # Trend filter: 1d close vs EMA50
        uptrend = close_1d_aligned[i] > ema_50_1d_aligned[i]
        downtrend = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # Bollinger Band breakout signals
        breakout_up = close[i] > upper_bb[i] and close[i-1] <= upper_bb[i-1]  # Cross above upper band
        breakout_down = close[i] < lower_bb[i] and close[i-1] >= lower_bb[i-1]  # Cross below lower band
        
        # Exit conditions
        exit_long = close[i] < middle_bb[i]  # Return to middle band
        exit_short = close[i] > middle_bb[i]  # Return to middle band
        exit_opposite_long = close[i] < lower_bb[i]  # Touch lower band while long
        exit_opposite_short = close[i] > upper_bb[i]  # Touch upper band while short
        
        # Trading logic
        if vol_confirm:
            # Long: BB breakout above upper band in uptrend
            if breakout_up and uptrend:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            # Short: BB breakout below lower band in downtrend
            elif breakout_down and downtrend:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25
            else:
                # Check for exits
                if position == 1 and (exit_long or exit_opposite_long):
                    position = 0
                    signals[i] = 0.0
                elif position == -1 and (exit_short or exit_opposite_short):
                    position = 0
                    signals[i] = 0.0
                else:
                    # Maintain current position
                    signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:
            # No volume confirmation: exit any position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals
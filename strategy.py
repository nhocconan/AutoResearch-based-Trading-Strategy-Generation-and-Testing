#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h volume spike and 1d trend filter
# - Enter long when 6h close breaks above Camarilla R4 (1d) AND 12h volume > 1.8x 20-period volume SMA AND 1d close > 1d EMA50
# - Enter short when 6h close breaks below Camarilla S4 (1d) AND 12h volume > 1.8x 20-period volume SMA AND 1d close < 1d EMA50
# - Exit: 6h close crosses below Camarilla PP (pivot point) for longs or above PP for shorts
# - Camarilla levels from 1d provide institutional support/resistance
# - Volume spike confirms breakout validity with participation
# - 1d EMA50 filter avoids counter-trend trades in ranging markets
# - Target: 12-25 trades/year to minimize fee drag while capturing high-probability breakouts

name = "6h_12h_1d_camarilla_breakout_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for Camarilla levels and trend filter (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Load 12h data ONCE before loop for volume confirmation (MTF rule compliance)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return signals
    
    # Pre-compute 1d OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for 1d: based on previous day's range
    # Camarilla formulas:
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # R2 = close + ((high - low) * 1.1/6)
    # R1 = close + ((high - low) * 1.1/12)
    # PP = (high + low + close) / 3
    # S1 = close - ((high - low) * 1.1/12)
    # S2 = close - ((high - low) * 1.1/6)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    daily_range = high_1d - low_1d
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    camarilla_r4 = close_1d + (daily_range * 1.1 / 2.0)
    camarilla_s4 = close_1d - (daily_range * 1.1 / 2.0)
    
    # Align Camarilla levels to 6h timeframe (using completed 1d bar)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Pre-compute EMA50 for 1d close (trend filter)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 1d close aligned for trend comparison
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Pre-compute volume SMA for 12h data (20-period)
    volume_12h = df_12h['volume'].values
    volume_sma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_sma_20_12h)
    
    for i in range(50, n):  # Start after 50-bar warmup for 50-period EMA and 20-period volume SMA
        # Skip if any required data is invalid
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(close_1d_aligned[i]) or np.isnan(volume_sma_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 1.8x 20-period volume SMA
        volume_12h_current = df_12h['volume'].values
        volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h_current)
        vol_confirm = volume_12h_aligned[i] > 1.8 * volume_sma_20_12h_aligned[i]
        
        # Trend filter: 1d close vs EMA50
        uptrend = close_1d_aligned[i] > ema_50_1d_aligned[i]
        downtrend = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # Camarilla breakout signals
        breakout_long = close[i] > camarilla_r4_aligned[i]  # Break above R4
        breakout_short = close[i] < camarilla_s4_aligned[i]  # Break below S4
        
        # Exit signals: close crosses pivot point (PP)
        exit_long = close[i] < camarilla_pp_aligned[i]   # Exit long when price falls below PP
        exit_short = close[i] > camarilla_pp_aligned[i]  # Exit short when price rises above PP
        
        # Trading logic
        if breakout_long and vol_confirm and uptrend:
            if position != 1:  # Only signal on new long entry
                position = 1
                signals[i] = 0.25
            else:
                signals[i] = 0.25
        elif breakout_short and vol_confirm and downtrend:
            if position != -1:  # Only signal on new short entry
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = -0.25
        else:
            # Check for mean reversion exits at pivot point
            if position == 1 and exit_long:
                position = 0
                signals[i] = 0.0
            elif position == -1 and exit_short:
                position = 0
                signals[i] = 0.0
            else:
                # Maintain current position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
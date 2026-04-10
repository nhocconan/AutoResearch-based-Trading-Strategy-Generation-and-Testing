#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla H3/L3 breakout with 1d EMA50 trend filter and volume confirmation
# - Uses 4h Camarilla H3/L3 as breakout levels from previous 4h candle (more responsive than 1d)
# - 1d EMA50 trend filter ensures trades align with medium-term trend (adapts to bull/bear)
# - Volume confirmation: current volume > 1.5x 20-period average to filter weak breakouts
# - Session filter: only trade 08-20 UTC to avoid low-volume Asian session noise
# - Exit: touch of opposite Camarilla level (L3/H3) or extreme levels (H4/L4) for reversal
# - Position size: 0.20 (20% of capital) to minimize fee drag
# - Target: 15-37 trades/year on 1h (60-150 total over 4 years) to stay within trade limits
# - Works in bull/bear: EMA50 reacts faster than EMA200 to regime changes, volume reduces false signals

name = "1h_4h_1d_camarilla_breakout_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA50 for trend filter (more responsive than EMA200)
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute 4h Camarilla levels (based on previous 4h candle's OHLC)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla calculation: based on previous 4h range
    rang = high_4h - low_4h
    camarilla_h4 = close_4h + 1.5 * rang * 1.1 / 2
    camarilla_l4 = close_4h - 1.5 * rang * 1.1 / 2
    camarilla_h3 = close_4h + 1.25 * rang * 1.1 / 2
    camarilla_l3 = close_4h - 1.25 * rang * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    h4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l4)
    
    # Pre-compute 1h volume average (20-period)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup for EMA50
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(trend_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        # Get current 4h close for trend filter (aligned)
        close_4h_current = df_4h['close'].values
        close_4h_aligned = align_htf_to_ltf(prices, df_4h, close_4h_current)
        
        # Get current 1d close for trend filter (aligned)
        close_1d_current = df_1d['close'].values
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d_current)
        
        # 1d trend filter: price > EMA50 = bullish, price < EMA50 = bearish
        bullish_trend = (not np.isnan(close_1d_aligned[i]) and not np.isnan(trend_aligned[i]) and 
                        close_1d_aligned[i] > trend_aligned[i])
        bearish_trend = (not np.isnan(close_1d_aligned[i]) and not np.isnan(trend_aligned[i]) and 
                        close_1d_aligned[i] < trend_aligned[i])
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price > H3 AND bullish trend AND volume confirmation
            if (prices['close'].iloc[i] > h3_aligned[i] and bullish_trend and volume_confirm):
                position = 1
                signals[i] = 0.20
            # Short conditions: price < L3 AND bearish trend AND volume confirmation
            elif (prices['close'].iloc[i] < l3_aligned[i] and bearish_trend and volume_confirm):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or reversal
            # Exit conditions: price touches opposite Camarilla level
            exit_long = prices['close'].iloc[i] < l3_aligned[i]   # Price breaks below L3 (exit long)
            exit_short = prices['close'].iloc[i] > h3_aligned[i]  # Price breaks above H3 (exit short)
            
            # Reversal conditions: price hits extreme levels (H4/L4) - counter-trend exit
            reverse_long = prices['close'].iloc[i] >= h4_aligned[i]  # Price hits H4 (reverse long)
            reverse_short = prices['close'].iloc[i] <= l4_aligned[i]  # Price hits L4 (reverse short)
            
            exit_condition = (position == 1 and (exit_long or reverse_long)) or \
                           (position == -1 and (exit_short or reverse_short))
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
    
    return signals
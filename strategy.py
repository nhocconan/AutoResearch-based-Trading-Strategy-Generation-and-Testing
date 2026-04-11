#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_camarilla_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Calculate daily Camarilla pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and levels
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels (standard multipliers)
    r4_1d = close_1d + range_1d * 1.1 / 2
    r3_1d = close_1d + range_1d * 1.1 / 4
    r2_1d = close_1d + range_1d * 1.1 / 6
    r1_1d = close_1d + range_1d * 1.1 / 12
    s1_1d = close_1d - range_1d * 1.1 / 12
    s2_1d = close_1d - range_1d * 1.1 / 6
    s3_1d = close_1d - range_1d * 1.1 / 4
    s4_1d = close_1d - range_1d * 1.1 / 2
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return signals
    
    # Weekly EMA50 for trend direction
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily pivots to 6h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Align weekly EMA to 6h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        r4 = r4_1d_aligned[i]
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        s4 = s4_1d_aligned[i]
        ema_trend = ema_50_1w_aligned[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma_20[i]
        
        # Trend filter: price relative to weekly EMA50
        above_trend = price_close > ema_trend
        below_trend = price_close < ema_trend
        
        # Camarilla-based signals
        long_signal = False
        short_signal = False
        
        # Long: price breaks above R4 with volume and above weekly trend
        if price_close > r4 and volume_confirmed and above_trend:
            long_signal = True
        
        # Short: price breaks below S4 with volume and below weekly trend
        if price_close < s4 and volume_confirmed and below_trend:
            short_signal = True
        
        # Mean reversion fades at R3/S3 (optional - commented out for now)
        # if price_close < r3 and price_close > s3 and volume_confirmed:
        #     if price_close < (r3 + s3) / 2 and above_trend:  # bias long in uptrend
        #         long_signal = True
        #     elif price_close > (r3 + s3) / 2 and below_trend:  # bias short in downtrend
        #         short_signal = True
        
        # Exit conditions: return to pivot or opposite extreme
        pivot_1d_val = (df_1d['high'].iloc[-1] + df_1d['low'].iloc[-1] + df_1d['close'].iloc[-1]) / 3 if len(df_1d) > 0 else 0
        pivot_aligned = align_htf_to_ltf(prices, df_1d, 
                                       np.full_like(df_1d['high'].values, pivot_1d_val))[i] if len(df_1d) > 0 else price_close
        
        exit_long = price_close < pivot_aligned
        exit_short = price_close > pivot_aligned
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 6s Camarilla pivot strategy with weekly EMA50 trend filter and volume confirmation.
# Enters long when price breaks above R4 (strong bullish breakout) with volume confirmation and above weekly EMA50 trend.
# Enters short when price breaks below S4 (strong bearish breakdown) with volume confirmation and below weekly EMA50 trend.
# Exits when price returns to daily pivot point (mean reversion to equilibrium).
# Uses weekly EMA50 to filter trades in direction of higher timeframe trend.
# Volume confirmation (>1.5x 20-period average) ensures institutional participation in breakouts.
# Target: 20-40 trades per year to minimize fee decay while capturing strong directional moves.
# Works in both bull and bear markets by trading breakouts in direction of weekly trend.
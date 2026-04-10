#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d ADX trend filter and volume confirmation
# - Primary: 6h price breaking above/below Camarilla R3/S3 levels from prior 1d session
# - HTF: 1d ADX(14) > 25 for trending market regime (only trade in strong trends)
# - HTF: 1d volume confirmation (current volume > 1.5x 20-period MA)
# - Long: 6h close > R3 + ADX>25 + volume confirmation
# - Short: 6h close < S3 + ADX>25 + volume confirmation
# - Exit: Price returns to Camarilla pivot point (PP) or opposite extreme (S4/R4)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: ADX filter ensures we only trade strong trends, Camarilla levels provide
#   precise entry/exit points, volume avoids false breakouts
# - Target: 80-160 trades over 4 years (20-40/year) to stay within fee drag limits

name = "6h_1d_camarilla_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 6h data
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels from prior 1d session
    # PP = (H + L + C) / 3
    # R3 = PP + (H - L) * 1.1 / 2
    # S3 = PP - (H - L) * 1.1 / 2
    # R4 = PP + (H - L) * 1.1
    # S4 = PP - (H - L) * 1.1
    PP_1d = (high_1d + low_1d + close_1d) / 3.0
    R3_1d = PP_1d + (high_1d - low_1d) * 1.1 / 2.0
    S3_1d = PP_1d - (high_1d - low_1d) * 1.1 / 2.0
    R4_1d = PP_1d + (high_1d - low_1d) * 1.1
    S4_1d = PP_1d - (high_1d - low_1d) * 1.1
    Pivot_PP_1d = PP_1d  # For exit condition
    
    # Align Camarilla levels to 6h timeframe (use prior day's levels)
    PP_1d_aligned = align_htf_to_ltf(prices, df_1d, PP_1d)
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    R4_1d_aligned = align_htf_to_ltf(prices, df_1d, R4_1d)
    S4_1d_aligned = align_htf_to_ltf(prices, df_1d, S4_1d)
    Pivot_PP_1d_aligned = align_htf_to_ltf(prices, df_1d, Pivot_PP_1d)
    
    # Calculate 1d ADX(14) for trend filter
    # ADX calculation: +DI, -DI, then DX, then smoothed ADX
    high_1d_shift = np.roll(high_1d, 1)
    low_1d_shift = np.roll(low_1d, 1)
    high_1d_shift[0] = high_1d[0]
    low_1d_shift[0] = low_1d[0]
    
    up_move = high_1d - high_1d_shift
    down_move = low_1d_shift - low_1d
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr_1d
    
    # DX and ADX
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    dx_1d = np.where((plus_di_1d + minus_di_1d) == 0, 0, dx_1d)
    adx_1d = pd.Series(dx_1d).ewm(alpha=1/14, adjust=False).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i]) or
            np.isnan(R3_1d_aligned[i]) or np.isnan(S3_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume (aligned to 6h)
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        
        # Regime conditions
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_1d_aligned[i] > 25.0
        
        # Volume confirmation: current 1d volume > 1.5x 20-period MA
        volume_confirm = volume_1d_aligned[i] > 1.5 * volume_ma_20_1d_aligned[i]
        
        # Camarilla breakout conditions
        breakout_long = close_6h[i] > R3_1d_aligned[i]
        breakout_short = close_6h[i] < S3_1d_aligned[i]
        
        # Exit conditions: return to pivot or opposite extreme
        exit_long = close_6h[i] < Pivot_PP_1d_aligned[i]  # Return to pivot
        exit_short = close_6h[i] > Pivot_PP_1d_aligned[i]  # Return to pivot
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Breakout above R3 + strong trend + volume confirmation
            if (breakout_long and strong_trend and volume_confirm):
                position = 1
                signals[i] = 0.25
            # Short entry: Breakout below S3 + strong trend + volume confirmation
            elif (breakout_short and strong_trend and volume_confirm):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price returns to pivot point
            if position == 1:  # Long position
                if exit_long:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if exit_short:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals
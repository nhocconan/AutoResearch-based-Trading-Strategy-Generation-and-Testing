#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Camarilla pivot levels provide precise intraday support/resistance derived from prior day's range.
# R3/S3 levels act as strong breakout/breakdown zones with institutional relevance.
# EMA34 on 1d ensures alignment with daily trend (bullish for longs above EMA, bearish for shorts below EMA).
# Volume spike confirms participation and reduces false breakouts.
# Designed for 20-40 trades/year on 4h to minimize fee drag and improve test generalization.
# Works in bull markets via trend-following breakouts and in bear markets via shorting breakdowns.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Need at least 1 bar for prior day's Camarilla
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels from prior 1d bar (using prior day's OHLC)
        if i >= 1:  # Ensure we have prior day's data
            # Get prior day's OHLC from 1d data (index i-1 in 1d corresponds to prior completed day)
            # Since we're on 4h timeframe, we need to map to 1d index
            # Use the aligned 1d data: we can use the prior completed 1d bar's values
            # We'll use close of prior 1d bar as reference, but need full OHLC
            # Instead, we compute Camarilla using the 1d data series directly
            # For simplicity and to avoid look-ahead, we use the 1d bar that closed prior to current 4h bar
            # Since we have df_1d, we can use its values
            # But we need to align: we want the Camarilla levels from the 1d bar that is fully completed
            # We'll use the 1d index: current 4h bar's time, find the prior 1d bar
            # However, to avoid complexity and look-ahead, we use a simpler approach:
            # Use the prior 4h bar's high/low as proxy? Not accurate.
            # Correct way: we need the prior day's OHLC. We can get it from df_1d.
            # Since df_1d is not aligned to 4h, we find the index of the 1d bar that ended before current 4h bar
            # But we already have align_htf_to_ltf for values, not for OHLC.
            # Alternative: compute Camarilla on 1d and align each level.
            # Given constraints, we compute prior day's OHLC from df_1d using the last completed 1d bar.
            # We can do: prior_1d_idx = min(len(df_1d)-1, (i // 6)) but that's manual MTF - not allowed.
            # Instead, we shift our approach: use the current 1d bar's OHLC to compute Camarilla for next day?
            # That would be look-ahead.
            # Simpler and valid: use the 1d bar's OHLC that is known at the close of the 1d bar.
            # For a 4h bar at time t, the relevant prior day is the day before the day containing t.
            # We can use: we are allowed to use data up to i, so we can use the 1d bar that ended at or before the 4h bar's day.
            # To avoid look-ahead and manual mapping, we compute the Camarilla levels on the 1d series and then align.
            # We'll compute: for each 1d bar, compute its Camarilla levels, then align to 4h.
            # This means the Camarilla levels from a 1d bar are available only after that 1d bar closes.
            # We'll do this outside the loop.
            pass  # We'll move Camarilla calculation outside the loop
        
        # Due to complexity of mapping prior day's OHLC in MTF without look-ahead,
        # we simplify: use the current 4h bar's high/low to compute intraday Camarilla-like levels?
        # But that's not standard.
        # Instead, we abandon Camarilla for this implementation and use Donchian as originally intended,
        # but with the proven winning pattern from the research.
        # Let's revert to a simpler, proven pattern: 4h Donchian(20) breakout with 1d EMA50 and volume.
        # But we saw that had 0 trades in history.
        # Why? Because the conditions were too tight.
        # Let's adjust: remove the EMA condition direction check and just use EMA as trend filter (price > EMA for long, etc)
        # And increase volume spike threshold to reduce trades.
        # But given the instruction to use Camarilla, and the fact that we're struggling with MTF mapping,
        # we note that the top performers use Camarilla with 1d EMA and volume.
        # We must implement Camarilla correctly.
        # We decide to compute the Camarilla levels for the 1d timeframe and align them.
        # For each 1d bar, we compute:
        #   H = df_1d['high'], L = df_1d['low'], C = df_1d['close']
        #   R3 = C + (H - L) * 1.1/4
        #   S3 = C - (H - L) * 1.1/4
        #   R4 = C + (H - L) * 1.1/2
        #   S4 = C - (H - L) * 1.1/2
        # But we only need R3 and S3 for breakout/breakdown.
        # We'll compute these on 1d and align to 4h.
        # This way, at a 4h bar, we get the Camarilla levels from the prior 1d bar (since the 1d bar must close first).
        # We'll do this before the loop.
        
        # We break out of this loop to compute Camarilla properly before the loop.
        break
    
    # If we broke out, we need to compute Camarilla before the loop.
    # Let's restart the function with proper MTF Camarilla calculation.
    
    # Re-enter: compute all indicators before loop
    
    # Session filter already computed
    
    # Get 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA34 for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Camarilla R3 and S3 on 1d: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    camarilla_r3_1d = c_1d + (h_1d - l_1d) * 1.1 / 4
    camarilla_s3_1d = c_1d - (h_1d - l_1d) * 1.1 / 4
    # Align Camarilla levels to 4h
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # Volume: 20-period EMA on 4h
    if n >= 20:
        vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    else:
        vol_ema_20 = volume.copy()
    volume_spike = volume > (2.0 * vol_ema_20)  # Threshold increased to reduce trades
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after EMA and volume warmup
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions using Camarilla R3/S3
        breakout_up = close[i] > camarilla_r3_aligned[i]
        breakout_down = close[i] < camarilla_s3_aligned[i]
        
        if position == 0:
            # Long: bullish breakout above R3 in 1d uptrend (price > EMA) with volume spike
            if breakout_up and close[i] > ema_34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakdown below S3 in 1d downtrend (price < EMA) with volume spike
            elif breakout_down and close[i] < ema_34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below R3 or loses 1d uptrend
            if close[i] < camarilla_r3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above S3 or loses 1d downtrend
            if close[i] > camarilla_s3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
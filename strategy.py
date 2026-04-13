#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    """
    Strategy: 6h Camarilla breakout with 1d ATR volume spike and 1d ADX trend filter
    Hypothesis: Camarilla breakouts work best when combined with volatility expansion (volume spike)
                and strong intraday trend (ADX > 25). This avoids false breakouts in low volatility,
                ranging markets. Discrete sizing (0.25) limits fee drag and drawdown.
                Target: 12-37 trades/year (50-150 over 4 years) to stay within 6h optimal range.
    """
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels, ATR volume spike, and ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla levels (based on previous day)
    # Pivot = (H+L+C)/3
    # H3 = Pivot + 1.1*(H-L)
    # L3 = Pivot - 1.1*(H-L)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    hl_range = high_1d - low_1d
    h3 = pivot + 1.1 * hl_range
    l3 = pivot - 1.1 * hl_range
    
    # Calculate 1d ATR(14) for volume spike confirmation
    tr_1d = np.maximum(
        high_1d - low_1d,
        np.maximum(
            np.abs(high_1d - np.roll(close_1d, 1)),
            np.abs(low_1d - np.roll(close_1d, 1))
        )
    )
    tr_1d[0] = high_1d[0] - low_1d[0]  # First bar
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume spike: current 1d volume > 2.0 x ATR(14) (proxy for volatility expansion)
    vol_spike_1d = volume_1d > (2.0 * atr_14_1d)
    
    # Calculate 1d ADX(14) for trend strength filter
    # +DM = max(High - Prev High, 0) if High - Prev High > Prev Low - Low else 0
    # -DM = max(Prev Low - Low, 0) if Prev Low - Low > High - Prev High else 0
    high_diff = np.diff(high_1d, prepend=high_1d[0])
    low_diff = -np.diff(low_1d, prepend=low_1d[0])  # Negative because we want positive values when low decreases
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # Wilder's smoothing for TR, +DM, -DM
    tr_rma = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False).mean().values
    plus_dm_rma = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_rma = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_rma / tr_rma
    minus_di = 100 * minus_dm_rma / tr_rma
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align all indicators to 6h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    # Calculate ATR using true range approximation for 6h timeframe
    atr_6h = np.zeros(n)
    for i in range(1, n):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        if i < 14:
            atr_6h[i] = tr  # Simple average for warmup
        else:
            atr_6h[i] = 0.93 * atr_6h[i-1] + 0.07 * tr  # Wilder's smoothing
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike confirmation: 1d volatility expansion
        volume_confirmed = bool(vol_spike_1d_aligned[i])
        
        # Trend filter: 1d ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Breakout conditions: price breaks Camarilla levels with volume spike and strong trend
        breakout_long = (close[i] > h3_aligned[i]) and volume_confirmed and strong_trend
        breakout_short = (close[i] < l3_aligned[i]) and volume_confirmed and strong_trend
        
        # Stoploss: 2x ATR below/above entry
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - 2.0 * atr_6h[i]
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + 2.0 * atr_6h[i]
        
        # Execute signals
        if breakout_long and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            elif position == -1:
                signals[i] = -position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            else:
                signals[i] = 0.0
                entry_price[i] = np.nan
    
    return signals

name = "6h_1d_camarilla_volspike_adx_v1"
timeframe = "6h"
leverage = 1.0
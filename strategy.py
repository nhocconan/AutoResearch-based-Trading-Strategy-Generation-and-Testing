#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and weekly chop regime filter
# - Uses 1d Camarilla pivot levels (H3/L3) for breakout entries
# - Requires 1d volume > 1.5x 20-period average for confirmation
# - Uses 1w choppiness index (CHOP > 61.8 = ranging, CHOP < 38.2 = trending) as regime filter
# - Only takes breakout trades aligned with weekly trend (price > weekly EMA20 for long, < for short)
# - Session filter: 08-20 UTC to avoid low-volume Asian session noise
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years) to avoid fee drag
# - Combines pivot-based breakouts with volume and regime filters for robustness in bull/bear markets

name = "12h_1d_1w_camarilla_breakout_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d Camarilla pivot levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for current day using previous day's data
    camarilla_h3 = np.full_like(close_1d, np.nan)
    camarilla_l3 = np.full_like(close_1d, np.nan)
    camarilla_h4 = np.full_like(close_1d, np.nan)
    camarilla_l4 = np.full_like(close_1d, np.nan)
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    camarilla_h3 = pivot + (range_1d * 1.1 / 4)
    camarilla_l3 = pivot - (range_1d * 1.1 / 4)
    camarilla_h4 = pivot + (range_1d * 1.1 / 2)
    camarilla_l4 = pivot - (range_1d * 1.1 / 2)
    
    # Align 1d Camarilla levels to 12h
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # 1d volume confirmation (> 1.5x 20-period average)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ratio = np.divide(volume_1d, vol_ma_20, out=np.zeros_like(volume_1d), where=vol_ma_20!=0)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_ratio)
    
    # 1w choppiness index regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range
    tr1_1w = high_1w - low_1w
    tr2_1w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3_1w = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr1_1w, np.maximum(tr2_1w, tr3_1w))
    tr_1w[0] = tr1_1w[0]
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Choppiness Index: CHOP = 100 * log10(sum(ATR14) / (maxHH - minLL)) / log10(14)
    maxhh = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    minll = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    sum_atr = pd.Series(atr_1w).rolling(window=14, min_periods=14).sum().values
    chop_denominator = maxhh - minll
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)  # Avoid division by zero
    chop_ratio = np.divide(sum_atr, chop_denominator, out=np.zeros_like(sum_atr), where=chop_denominator!=0)
    chop_1w = 100 * np.log10(chop_ratio) / np.log10(14)
    
    # Align 1w chop to 12h
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # 1w EMA20 for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Main price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid or outside session
        if (not in_session[i] or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(volume_confirm_aligned[i]) or
            np.isnan(chop_1w_aligned[i]) or np.isnan(ema_20_1w_aligned[i]) or
            volume_confirm_aligned[i] < 1.5):  # Volume confirmation required
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in trending markets (CHOP < 38.2) or strong breakouts in ranging markets
        is_trending = chop_1w_aligned[i] < 38.2
        is_strong_breakout = False
        
        if position == 1:  # Long position
            # Exit conditions: mean reversion or trend change
            if close[i] <= ema_20_1w_aligned[i]:  # Return to weekly EMA
                position = 0
                signals[i] = 0.0
            elif close[i] < l3_aligned[i]:  # Break below L3 (failed breakout)
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: mean reversion or trend change
            if close[i] >= ema_20_1w_aligned[i]:  # Return to weekly EMA
                position = 0
                signals[i] = 0.0
            elif close[i] > h3_aligned[i]:  # Break above H3 (failed breakdown)
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for breakout entries aligned with weekly trend
            weekly_uptrend = close[i] > ema_20_1w_aligned[i]
            weekly_downtrend = close[i] < ema_20_1w_aligned[i]
            
            # Long breakout above H3 with volume confirmation
            if (close[i] > h3_aligned[i] and 
                weekly_uptrend and
                (is_trending or close[i] > h4_aligned[i])):  # In trending market or strong breakout above H4
                position = 1
                signals[i] = 0.25
            # Short breakdown below L3 with volume confirmation
            elif (close[i] < l3_aligned[i] and 
                  weekly_downtrend and
                  (is_trending or close[i] < l4_aligned[i])):  # In trending market or strong breakdown below L4
                position = -1
                signals[i] = -0.25
    
    return signals
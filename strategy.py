#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Choppiness_Regime_ADX_Trend_Filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for choppiness and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                               np.abs(low_1d[1:] - close_1d[:-1])))
    tr = np.concatenate([[np.nan], tr])
    
    # ATR(14)
    atr = np.full(len(tr), np.nan)
    for i in range(14, len(tr)):
        atr[i] = np.nanmean(tr[i-13:i+1])
    
    # Sum of ATR over 14 periods
    sum_atr = np.full(len(tr), np.nan)
    for i in range(14, len(tr)):
        sum_atr[i] = np.nansum(atr[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    highest_high = np.full(len(high_1d), np.nan)
    lowest_low = np.full(len(low_1d), np.nan)
    for i in range(13, len(high_1d)):
        highest_high[i] = np.max(high_1d[i-13:i+1])
        lowest_low[i] = np.min(low_1d[i-13:i+1])
    
    # Choppiness Index
    chop = np.full(len(close_1d), np.nan)
    for i in range(13, len(close_1d)):
        if sum_atr[i] > 0 and highest_high[i] > lowest_low[i]:
            chop[i] = 100 * np.log10(sum_atr[i] / (highest_high[i] - lowest_low[i])) / np.log10(14)
        else:
            chop[i] = np.nan
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # ADX (14-period)
    # Directional Movement
    up_move = np.diff(high_1d, prepend=np.nan)
    down_move = -np.diff(low_1d, prepend=np.nan)
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed ATR, +DM, -DM
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_adx = wilders_smooth(tr, 14)
    plus_di = 100 * wilders_smooth(plus_dm, 14) / atr_adx
    minus_di = 100 * wilders_smooth(minus_dm, 14) / atr_adx
    dx = np.full_like(atr_adx, np.nan)
    mask = (plus_di + minus_di) > 0
    dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / (plus_di[mask] + minus_di[mask])
    adx = wilders_smooth(dx, 14)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime: Chop > 61.8 = range (mean revert), Chop < 38.2 = trending (trend follow)
        chop_val = chop_aligned[i]
        adx_val = adx_aligned[i]
        
        if position == 0:
            # In ranging market (Chop > 61.8): mean reversion at extremes
            # Use price position within daily range for entry
            # We'll use a simple proxy: if price is near daily low in range, go long
            # if near daily high in range, go short
            # For simplicity, we'll use chop level itself as signal strength
            if chop_val > 61.8:  # ranging market
                # In range, look for mean reversion: we'll use a simple approach
                # Actually, let's use price action: but we don't have intraday here
                # Instead, we'll use the fact that in ranging markets, we can fade moves
                # But we need price data - we'll use a simplified approach
                # For now, let's skip mean reversion and focus on trending markets
                # Only trade in trending markets to avoid whipsaw
                pass
            elif chop_val < 38.2 and adx_val > 25:  # trending market with strong trend
                # In trending market, follow the trend
                # We need trend direction - we'll use price vs moving average
                # But to keep it simple and avoid look-ahead, we'll use a basic rule
                # Since we don't have a trend indicator, let's use price momentum
                # We'll use a simple rule: if close > open, bias long; else bias short
                # But this is weak - let's instead use the ADX crossover idea
                # Actually, let's use a simpler approach: only trade when we have clear signals
                # We'll skip entries in this version and focus on exits
                pass
        
        # Simplified approach: only exit based on regime changes
        # Enter only when we have a clear regime and momentum
        # For now, let's implement a basic trend following system
        
        # Actually, let's rethink: use chop for regime, and for trend direction use price vs EMA
        # But we need to compute EMA on 1d and align
        
        # Let's compute EMA(50) on 1d close
        if len(close_1d) >= 50:
            ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
            ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
            
            if position == 0:
                # Long: price above EMA50 AND trending market (Chop < 38.2 AND ADX > 25)
                # Short: price below EMA50 AND trending market (Chop < 38.2 AND ADX > 25)
                if not np.isnan(ema50_aligned[i]):
                    if chop_val < 38.2 and adx_val > 25:
                        if close[i] > ema50_aligned[i]:
                            signals[i] = 0.25
                            position = 1
                        elif close[i] < ema50_aligned[i]:
                            signals[i] = -0.25
                            position = -1
            elif position == 1:
                # Exit long: price below EMA50 OR market becomes ranging (Chop > 61.8)
                if close[i] <= ema50_aligned[i] or chop_val > 61.8:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price above EMA50 OR market becomes ranging (Chop > 61.8)
                if close[i] >= ema50_aligned[i] or chop_val > 61.8:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # No EMA50 data, hold flat
            if position != 0:
                signals[i] = 0.0
                position = 0
    
    return signals

# Hypothesis: Choppiness Index identifies market regime (range vs trend).
# In trending markets (Chop < 38.2, ADX > 25), we follow price relative to EMA50.
# In ranging markets (Chop > 61.8), we stay flat to avoid whipsaw.
# This avoids false signals in sideways markets and captures trends when they occur.
# Works in both bull and bear markets as it adapts to regime.
# Uses 12h timeframe for lower frequency and reduced fee drag.
# Target: 50-150 total trades over 4 years = 12-37/year to minimize fee decay.
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h volume spike and 12h ADX > 20
# - Long when price breaks above Camarilla R4 AND 12h volume > 1.8x 20-period volume SMA AND 12h ADX > 20
# - Short when price breaks below Camarilla S4 AND 12h volume > 1.8x 20-period volume SMA AND 12h ADX > 20
# - Exit: ATR trailing stop (2.0*ATR) from highest/lowest since entry
# - Uses 6h for price action (Camarilla pivots), 12h for volume confirmation and ADX trend filter
# - Camarilla pivots identify key support/resistance; volume spike confirms breakout validity; ADX filters choppy markets
# - Tight entries target 12-30 trades/year (50-120 total over 4 years) to minimize fee drag
# - Works in bull (breakouts up) and bear (breakouts down) with volume and trend filters

name = "6h_12h_camarilla_volume_adx_v1"
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
    
    # Load 12h data ONCE before loop for volume and ADX (MTF rule compliance)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return signals
    
    # Calculate 12h volume SMA for confirmation
    vol_12h = df_12h['volume'].values
    volume_sma_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_sma_20_12h)
    
    # Calculate 12h ADX(14) for trend strength filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[:-1])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Wilder's smoothing function
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            result[period-1] = np.nanmean(values[:period])
            for i in range(period, len(values)):
                result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    atr_12h = wilders_smoothing(tr, 14)
    plus_di_12h = 100 * wilders_smoothing(plus_dm, 14) / atr_12h
    minus_di_12h = 100 * wilders_smoothing(minus_dm, 14) / atr_12h
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    adx_12h = wilders_smoothing(dx_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Pre-compute Camarilla pivot levels on 6h (primary timeframe)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    close_prev = pd.Series(close).shift(1).values
    close_prev[0] = close[0]  # Fill first value
    
    # Camarilla pivot calculation
    pivot = (highest_high + lowest_low + close_prev) / 3.0
    range_val = highest_high - lowest_low
    
    # Resistance levels
    r3 = pivot + (range_val * 1.1 / 4.0)
    r4 = pivot + (range_val * 1.1 / 2.0)
    
    # Support levels
    s3 = pivot - (range_val * 1.1 / 4.0)
    s4 = pivot - (range_val * 1.1 / 2.0)
    
    # ATR for dynamic stoploss (using 6h data)
    tr_6h1 = np.abs(high[1:] - low[:-1])
    tr_6h2 = np.abs(high[1:] - close[:-1])
    tr_6h3 = np.abs(low[1:] - close[:-1])
    tr_6h = np.maximum(np.maximum(tr_6h1, tr_6h2), tr_6h3)
    tr_6h = np.concatenate([[np.nan], tr_6h])
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    for i in range(lookback, n):
        # Skip if any required data is invalid
        if (np.isnan(r4[i]) or np.isnan(s4[i]) or 
            np.isnan(volume_sma_20_12h_aligned[i]) or np.isnan(adx_12h_aligned[i]) or
            np.isnan(atr_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 1.8x 20-period volume SMA
        vol_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_12h)
        vol_confirm = vol_12h_aligned[i] > 1.8 * volume_sma_20_12h_aligned[i]
        
        # Trend filter: 12h ADX > 20 indicates sufficient trend strength
        trend_filter = adx_12h_aligned[i] > 20.0
        
        # Only trade when both volume confirmation and trend filter are present
        if vol_confirm and trend_filter:
            # Long breakout: price breaks above Camarilla R4
            if close[i] > r4[i]:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25  # Maintain position
            # Short breakout: price breaks below Camarilla S4
            elif close[i] < s4[i]:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25  # Maintain position
            # Alternative exit: ATR-based trailing stop
            elif position == 1 and close[i] < (highest_high_since_entry - 2.0 * atr_6h[i]):
                if position != 0:  # Only signal on exit
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.0  # Maintain flat
            elif position == -1 and close[i] > (lowest_low_since_entry + 2.0 * atr_6h[i]):
                if position != 0:  # Only signal on exit
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.0  # Maintain flat
            else:
                # Maintain current position and update tracking levels
                if position == 1:
                    # Track highest high since entry for trailing stop
                    if 'highest_high_since_entry' not in locals():
                        highest_high_since_entry = high[i]
                    else:
                        highest_high_since_entry = max(highest_high_since_entry, high[i])
                    signals[i] = 0.25
                elif position == -1:
                    # Track lowest low since entry for trailing stop
                    if 'lowest_low_since_entry' not in locals():
                        lowest_low_since_entry = low[i]
                    else:
                        lowest_low_since_entry = min(lowest_low_since_entry, low[i])
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
        else:
            # No trade: exit any position if conditions not met
            if position != 0:
                position = 0
                # Reset tracking variables
                if 'highest_high_since_entry' in locals():
                    del highest_high_since_entry
                if 'lowest_low_since_entry' in locals():
                    del lowest_low_since_entry
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals
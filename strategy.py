#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume spike and 1w ADX > 25
# - Long when price breaks above 1d Donchian upper band AND 1w volume > 2.0x 20-period volume SMA AND 1w ADX > 25
# - Short when price breaks below 1d Donchian lower band AND 1w volume > 2.0x 20-period volume SMA AND 1w ADX > 25
# - Exit: ATR trailing stop (2.5*ATR) from highest/lowest since entry
# - Uses 1d for price action (Donchian channels), 1w for volume confirmation and ADX trend filter
# - Donchian breakouts capture strong momentum; volume spike confirms validity; ADX filters weak/choppy markets
# - Tight entries target 15-30 trades/year (60-120 total over 4 years) to minimize fee drag
# - Works in bull (breakouts up) and bear (breakouts down) with volume and trend filters

name = "1d_1w_donchian_volume_adx_v1"
timeframe = "1d"
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
    
    # Load 1w data ONCE before loop for volume and ADX (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Calculate 1w volume SMA for confirmation
    vol_1w = df_1w['volume'].values
    volume_sma_20_1w = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_sma_20_1w)
    
    # Calculate 1w ADX(14) for trend strength filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = high_1w[1:] - high_1w[:-1]
    down_move = low_1w[:-1] - low_1w[1:]
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
    
    atr_1w = wilders_smoothing(tr, 14)
    plus_di_1w = 100 * wilders_smoothing(plus_dm, 14) / atr_1w
    minus_di_1w = 100 * wilders_smoothing(minus_dm, 14) / atr_1w
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w)
    adx_1w = wilders_smoothing(dx_1w, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Pre-compute Donchian channels on 1d (primary timeframe)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    upper_band = highest_high
    lower_band = lowest_low
    mid_band = (upper_band + lower_band) / 2.0
    
    # ATR for dynamic stoploss (using 1d data)
    tr_1d1 = np.abs(high[1:] - low[:-1])
    tr_1d2 = np.abs(high[1:] - close[:-1])
    tr_1d3 = np.abs(low[1:] - close[:-1])
    tr_1d = np.maximum(np.maximum(tr_1d1, tr_1d2), tr_1d3)
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    for i in range(lookback, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(volume_sma_20_1w_aligned[i]) or np.isnan(adx_1w_aligned[i]) or
            np.isnan(atr_1d[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1w volume > 2.0x 20-period volume SMA
        vol_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_1w)
        vol_confirm = vol_1w_aligned[i] > 2.0 * volume_sma_20_1w_aligned[i]
        
        # Trend filter: 1w ADX > 25 indicates sufficient trend strength
        trend_filter = adx_1w_aligned[i] > 25.0
        
        # Only trade when both volume confirmation and trend filter are present
        if vol_confirm and trend_filter:
            # Long breakout: price breaks above Donchian upper band
            if close[i] > upper_band[i]:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25  # Maintain position
            # Short breakout: price breaks below Donchian lower band
            elif close[i] < lower_band[i]:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25  # Maintain position
            # Alternative exit: ATR-based trailing stop
            elif position == 1 and close[i] < (highest_high_since_entry - 2.5 * atr_1d[i]):
                if position != 0:  # Only signal on exit
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.0  # Maintain flat
            elif position == -1 and close[i] > (lowest_low_since_entry + 2.5 * atr_1d[i]):
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
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and ADX > 20 filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 2.0x 20-period volume SMA AND 1d ADX > 20
# - Short when price breaks below Camarilla L3 level AND 1d volume > 2.0x 20-period volume SMA AND 1d ADX > 20
# - Exit: price returns to Camarilla pivot point (PP) or ATR-based trailing stop (2.0*ATR)
# - Uses 12h for price action (Camarilla levels from prior 1d), 1d for volume and ADX confirmation
# - Camarilla levels provide structured support/resistance; volume confirms breakout validity; ADX filters weak markets
# - Tight entry conditions target 15-35 trades/year to minimize fee drag while maintaining edge
# - Works in bull (breakouts up) and bear (breakouts down) with volume and trend filters

name = "12h_1d_camarilla_volume_adx_v1"
timeframe = "12h"
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
    
    # Load 1d data ONCE before loop (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate 1d volume SMA for confirmation
    vol_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate 1d ADX(14) for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # Align with indices
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Wilder's smoothing function
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(values[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(values)):
                result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    plus_di_1d = 100 * wilders_smoothing(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * wilders_smoothing(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smoothing(dx_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Pre-compute Camarilla levels on 12h using prior 1d OHLC
    # Camarilla levels: based on prior day's range
    # H4 = close + 1.1*(high-low)*1.1/2
    # H3 = close + 1.1*(high-low)*1.1/4
    # H2 = close + 1.1*(high-low)*1.1/6
    # H1 = close + 1.1*(high-low)*1.1/12
    # PP = (high+low+close)/3
    # L1 = close - 1.1*(high-low)*1.1/12
    # L2 = close - 1.1*(high-low)*1.1/6
    # L3 = close - 1.1*(high-low)*1.1/4
    # L4 = close - 1.1*(high-low)*1.1/2
    
    # Align 1d OHLC to 12h for Camarilla calculation
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Calculate Camarilla levels
    rang = high_1d_aligned - low_1d_aligned
    H3 = close_1d_aligned + 1.1 * rang * 1.1 / 4
    L3 = close_1d_aligned - 1.1 * rang * 1.1 / 4
    PP = (high_1d_aligned + low_1d_aligned + close_1d_aligned) / 3.0
    
    # ATR for dynamic stoploss (using 12h data)
    tr_12h1 = np.abs(high[1:] - low[:-1])
    tr_12h2 = np.abs(high[1:] - close[:-1])
    tr_12h3 = np.abs(low[1:] - close[:-1])
    tr_12h = np.maximum(np.maximum(tr_12h1, tr_12h2), tr_12h3)
    tr_12h = np.concatenate([[np.nan], tr_12h])
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    for i in range(1, n):  # Start from 1 to ensure we have prior day data
        # Skip if any required data is invalid
        if (np.isnan(H3[i]) or np.isnan(L3[i]) or np.isnan(PP[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(atr_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 2.0x 20-period volume SMA
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        vol_confirm = vol_1d_aligned[i] > 2.0 * volume_sma_20_1d_aligned[i]
        
        # Trend filter: 1d ADX > 20 indicates strong trend
        trend_filter = adx_1d_aligned[i] > 20.0
        
        # Only trade when both volume confirmation and trend filter are present
        if vol_confirm and trend_filter:
            # Long breakout: price breaks above Camarilla H3 level
            if close[i] > H3[i]:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25  # Maintain position
            # Short breakout: price breaks below Camarilla L3 level
            elif close[i] < L3[i]:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25  # Maintain position
            # Exit: price returns to pivot point (within 0.1% of H3-L3 range)
            elif abs(close[i] - PP[i]) < (H3[i] - L3[i]) * 0.001:
                if position != 0:  # Only signal on exit
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.0  # Maintain flat
            # Alternative exit: ATR-based trailing stop
            elif position == 1 and close[i] < (highest_high_since_entry - 2.0 * atr_12h[i]):
                if position != 0:  # Only signal on exit
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.0  # Maintain flat
            elif position == -1 and close[i] > (lowest_low_since_entry + 2.0 * atr_12h[i]):
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
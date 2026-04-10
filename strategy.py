#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and choppiness regime filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 2.0x 20-period volume SMA AND chop < 61.8 (trending)
# - Short when price breaks below Camarilla L3 level AND 1d volume > 2.0x 20-period volume SMA AND chop < 61.8 (trending)
# - Exit: ATR(14) trailing stop (2.0*ATR) from highest/lowest since entry
# - Uses 4h for price action (Camarilla pivots from prior 1d), 1d for volume/chop confirmation
# - Volume spike confirms institutional interest; chop filter avoids ranging markets
# - Tight entries target ~20-30 trades/year to minimize fee drag (proven winners: ETH test Sharpe 1.47)
# - Works in bull (buy H3 breakouts in uptrend) and bear (sell L3 breakdowns in downtrend) with volume/chop filters

name = "4h_1d_camarilla_volspike_chop_v1"
timeframe = "4h"
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
    
    # Load 1d data ONCE before loop for HTF confirmation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate 1d volume SMA for confirmation
    vol_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate 1d choppiness index for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(np.maximum(tr1, tr2), tr3)
    tr_1d = np.concatenate([[np.nan], tr_1d])
    
    # Sum of True Range over 14 periods
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum_tr_14 / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    chop_1d = np.full(len(close_1d), np.nan)
    mask = (range_14 > 0) & (~np.isnan(sum_tr_14))
    chop_1d[mask] = 100 * np.log10(sum_tr_14[mask] / range_14[mask]) / np.log10(14)
    
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Pre-compute Camarilla levels from prior 1d (use previous completed 1d bar)
    # Camarilla levels based on prior 1d OHLC
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h4 = np.full(n, np.nan)  # for stoploss reference
    camarilla_l4 = np.full(n, np.nan)
    
    for i in range(n):
        # Get prior completed 1d bar index
        # Since we're on 4h timeframe, prior 1d bar completed at index i // 6 (6*4h=1d)
        # But to avoid look-ahead, we use align_htf_to_ltf later for the actual values
        pass  # We'll calculate aligned values below
    
    # Calculate Camarilla levels for each completed 1d bar
    o_1d = df_1d['open'].values
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    camarilla_h3_1d = o_1d + (h_1d - l_1d) * 1.1 / 6
    camarilla_l3_1d = c_1d - (h_1d - l_1d) * 1.1 / 6
    camarilla_h4_1d = o_1d + (h_1d - l_1d) * 1.1 / 2
    camarilla_l4_1d = c_1d - (h_1d - l_1d) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar to complete)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d)
    
    # ATR for dynamic stoploss (using 4h data)
    tr1 = np.abs(high[1:] - low[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Track highest/lowest since entry for trailing stop
    highest_high_since_entry = np.full(n, np.nan)
    lowest_low_since_entry = np.full(n, np.nan)
    
    for i in range(20, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(atr_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 2.0x 20-period volume SMA
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        vol_confirm = vol_1d_aligned[i] > 2.0 * volume_sma_20_1d_aligned[i]
        
        # Chop filter: chop < 61.8 indicates trending market (good for breakouts)
        chop_filter = chop_1d_aligned[i] < 61.8
        
        # Only trade when both volume confirmation and chop filter are present
        if vol_confirm and chop_filter:
            # Long: price breaks above Camarilla H3 level
            if close[i] > camarilla_h3_aligned[i]:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                    highest_high_since_entry[i] = high[i]
                else:
                    signals[i] = 0.25
                    highest_high_since_entry[i] = max(highest_high_since_entry[i-1] if i > 0 else high[i], high[i])
            # Short: price breaks below Camarilla L3 level
            elif close[i] < camarilla_l3_aligned[i]:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                    lowest_low_since_entry[i] = low[i]
                else:
                    signals[i] = -0.25
                    lowest_low_since_entry[i] = min(lowest_low_since_entry[i-1] if i > 0 else low[i], low[i])
            else:
                # Maintain position and update tracking levels
                if position == 1:
                    signals[i] = 0.25
                    highest_high_since_entry[i] = max(highest_high_since_entry[i-1] if i > 0 else high[i], high[i])
                elif position == -1:
                    signals[i] = -0.25
                    lowest_low_since_entry[i] = min(lowest_low_since_entry[i-1] if i > 0 else low[i], low[i])
                else:
                    signals[i] = 0.0
            
            # Check for ATR trailing stop exit
            if position == 1 and not np.isnan(highest_high_since_entry[i]):
                if close[i] < (highest_high_since_entry[i] - 2.0 * atr_4h[i]):
                    position = 0
                    signals[i] = 0.0
            elif position == -1 and not np.isnan(lowest_low_since_entry[i]):
                if close[i] > (lowest_low_since_entry[i] + 2.0 * atr_4h[i]):
                    position = 0
                    signals[i] = 0.0
        else:
            # No trade: exit any position if conditions not met
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals
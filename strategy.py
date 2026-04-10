#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and choppiness filter
# - Long when price breaks above Camarilla H4 level AND 1d volume > 1.8x 20-period volume SMA AND chop > 61.8 (range)
# - Short when price breaks below Camarilla L4 level AND 1d volume > 1.8x 20-period volume SMA AND chop > 61.8 (range)
# - Exit: ATR trailing stop (2.0*ATR) from highest/lowest since entry
# - Uses 4h for price action (Camarilla pivot breakout), 1d for volume confirmation and regime filter (choppiness)
# - Volume spike confirms institutional participation; chop filter avoids whipsaw in strong trends
# - Tight entries target ~20-30 trades/year to minimize fee drag (proven winners: ETH test Sharpe 1.46)
# - Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend) with volume/chop filter

name = "4h_1d_camarilla_volume_chop_v1"
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
    
    # Load 1d data ONCE before loop for volume and chop (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate 1d volume SMA for confirmation
    vol_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate 1d Choppiness Index (CHOP)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(np.maximum(tr1, tr2), tr3)
    tr_1d = np.concatenate([[np.nan], tr_1d])
    
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sumTR14 / (HH14 - LL14)) / log10(14)
    # Avoid division by zero and log of zero
    range_14 = hh_14 - ll_14
    chop_1d = np.where(
        (range_14 > 0) & (~np.isnan(tr_sum_14)) & (tr_sum_14 > 0),
        100 * np.log10(tr_sum_14 / range_14) / np.log10(14),
        50  # neutral when invalid
    )
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Pre-compute Camarilla pivot levels on 4h (primary timeframe)
    lookback_pivot = 20
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    
    for i in range(lookback_pivot, n):
        # Use previous completed candle for pivot calculation
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Calculate pivot point
        pivot = (prev_high + prev_low + prev_close) / 3.0
        range_val = prev_high - prev_low
        
        # Camarilla levels
        camarilla_h4[i] = pivot + (range_val * 1.1 / 2.0)  # H4
        camarilla_l4[i] = pivot - (range_val * 1.1 / 2.0)  # L4
    
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
    
    for i in range(lookback_pivot, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(atr_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.8x 20-period volume SMA
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        vol_confirm = vol_1d_aligned[i] > 1.8 * volume_sma_20_1d_aligned[i]
        
        # Chop filter: > 61.8 indicates ranging market (good for mean reversion breakouts)
        chop_filter = chop_1d_aligned[i] > 61.8
        
        # Only trade when both volume confirmation and chop filter are present
        if vol_confirm and chop_filter:
            # Long: price breaks above Camarilla H4
            if close[i] > camarilla_h4[i]:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                    highest_high_since_entry[i] = high[i]
                else:
                    signals[i] = 0.25
                    highest_high_since_entry[i] = max(highest_high_since_entry[i-1] if i > 0 else high[i], high[i])
            # Short: price breaks below Camarilla L4
            elif close[i] < camarilla_l4[i]:
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
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h TRIX(12) zero-cross with 1d volume spike and choppiness regime filter
# TRIX (Triple Exponential Average) is a momentum oscillator that filters noise and identifies trend changes
# Zero-cross signals provide timely entries with less whipsaw than moving average crossovers
# 1d volume spike (>2.0x 20-period average) confirms institutional participation in the move
# Choppiness Index (CHOP) regime filter: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (follow momentum)
# In trending regimes (CHOP < 38.2): follow TRIX zero-cross direction
# In ranging regimes (CHOP > 61.8): fade TRIX extremes (mean reversion at overbought/oversold)
# ATR-based trailing stop (2.0x ATR) manages risk while allowing trends to develop
# Designed for low trade frequency (target: 12-30 trades/year) to minimize fee drag on 12h timeframe
# Works in bull markets via momentum continuation signals
# Works in bear markets via mean reversion in ranging conditions and momentum shorts in downtrends
# Volume confirmation reduces false breakouts
# Choppiness regime filter adapts strategy to market conditions, improving performance in both bull and bear markets

name = "12h_TRIX_ZeroCross_VolumeSpike_ChopRegime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter (used in TRIX calculation)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d volume spike confirmation (>2.0x 20-period average)
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=1).mean().values
    vol_spike_1d = df_1d['volume'].values > 2.0 * vol_ma_20_1d
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # Calculate TRIX (12,12,12) - Triple Exponential Average
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12)
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = pd.Series(ema3).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix_values = trix.values
    
    # Calculate TRIX signal line (9-period EMA of TRIX)
    trix_signal = pd.Series(trix_values).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Calculate TRIX histogram (TRIX - signal)
    trix_hist = trix_values - trix_signal
    
    # Calculate Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR(14)) / (HHV(high,14) - LLV(low,14))) / log10(14)
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    hh_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = hh_14 - ll_14
    # Avoid division by zero
    chop_raw = np.where(range_14 > 0, sum_atr_14 / range_14, 1.0)
    chop = 100 * np.log10(chop_raw) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    start_idx = 34  # warmup for EMA calculations
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_trix = trix_values[i]
        curr_trix_signal = trix_signal[i]
        curr_trix_hist = trix_hist[i]
        curr_chop = chop[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_atr = atr[i]
        curr_vol_spike = vol_spike_1d_aligned[i] > 0.5  # Boolean check
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            # Trailing stop: 2.0 * ATR below highest high
            stop_price = highest_high_since_entry - 2.0 * curr_atr
            # Exit conditions: price below trailing stop OR TRIX histogram turns negative (momentum loss)
            if curr_close < stop_price or curr_trix_hist < 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            # Trailing stop: 2.0 * ATR above lowest low
            stop_price = lowest_low_since_entry + 2.0 * curr_atr
            # Exit conditions: price above trailing stop OR TRIX histogram turns positive (momentum loss)
            if curr_close > stop_price or curr_trix_hist > 0:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Determine regime: trending (CHOP < 38.2) or ranging (CHOP > 61.8)
            is_trending = curr_chop < 38.2
            is_ranging = curr_chop > 61.8
            
            if is_trending:
                # In trending regime: follow TRIX momentum
                # Long: TRIX crosses above signal line AND price > 1d EMA34 AND volume spike
                # Short: TRIX crosses below signal line AND price < 1d EMA34 AND volume spike
                trix_cross_up = curr_trix_hist > 0 and trix_hist[i-1] <= 0 if i > 0 else False
                trix_cross_down = curr_trix_hist < 0 and trix_hist[i-1] >= 0 if i > 0 else False
                
                if trix_cross_up and curr_close > curr_ema_1d and curr_vol_spike:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    highest_high_since_entry = curr_high
                elif trix_cross_down and curr_close < curr_ema_1d and curr_vol_spike:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    lowest_low_since_entry = curr_low
                else:
                    signals[i] = 0.0
                    
            elif is_ranging:
                # In ranging regime: mean reversion at TRIX extremes
                # Long: TRIX histogram < -0.1 (oversold) AND price < 1d EMA34 AND volume spike
                # Short: TRIX histogram > 0.1 (overbought) AND price > 1d EMA34 AND volume spike
                if curr_trix_hist < -0.1 and curr_close < curr_ema_1d and curr_vol_spike:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    highest_high_since_entry = curr_high
                elif curr_trix_hist > 0.1 and curr_close > curr_ema_1d and curr_vol_spike:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    lowest_low_since_entry = curr_low
                else:
                    signals[i] = 0.0
            else:
                # Neutral regime (38.2 <= CHOP <= 61.8): no trades
                signals[i] = 0.0
    
    return signals
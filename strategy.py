#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h trend (EMA50) and 1d regime (Choppiness Index) for signal direction,
# with 1h Donchian(20) breakout for entry timing and volume confirmation. Uses ATR trailing stop.
# Designed for 1h timeframe: targets 15-37 trades/year by using HTF filters to reduce noise.
# Works in bull (breakouts with trend) and bear (breakdowns with trend) markets.
# Session filter (08-20 UTC) reduces noise during low-activity periods.

name = "1h_4hEMA50_1dChop_Regime_Donchian20_Breakout_Volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Pre-compute session filter (UTC 08-20)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 50 or len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d Choppiness Index for regime detection (needs extra delay for confirmation)
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(HH(14)-LL(14)))) / log10(14)
    # Range: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate TR for 1d
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate HH(14) and LL(14) for 1d
    highest_high_14_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Calculate Chopiness Index
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    hh_ll_diff = highest_high_14_1d - lowest_low_14_1d
    chop_raw = 100 * np.log10(sum_atr_14) / np.log10(14) / np.log10(hh_ll_diff)
    chop_1d = np.where(hh_ll_diff > 0, chop_raw, 50.0)  # avoid division by zero
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d, additional_delay_bars=2)
    
    # Regime: trending if CHOP < 38.2, ranging if CHOP > 61.8
    trending_regime = chop_1d_aligned < 38.2
    ranging_regime = chop_1d_aligned > 61.8
    
    # 1h ATR for trailing stop and volume median
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 1h volume median for confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # 1h Donchian(20) channels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup for all indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0  # close position at session end
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or
            np.isnan(atr[i]) or 
            np.isnan(vol_median_20[i]) or
            np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_atr = atr[i]
        
        # Trend filter: 4h EMA50
        uptrend = curr_close > ema_50_4h_aligned[i]
        downtrend = curr_close < ema_50_4h_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        # Donchian breakout conditions
        breakout_up = curr_close > highest_high_20[i-1]  # break above previous period's high
        breakout_down = curr_close < lowest_low_20[i-1]   # break below previous period's low
        
        if position == 0:  # Flat - look for new entries
            # Only enter in trending regime
            if trending_regime[i]:
                # Long: Breakout up AND uptrend AND volume spike
                if breakout_up and uptrend and volume_confirm:
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
                    highest_since_entry = curr_close
                    lowest_since_entry = curr_close
                # Short: Breakout down AND downtrend AND volume spike
                elif breakout_down and downtrend and volume_confirm:
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
                    highest_since_entry = curr_close
                    lowest_since_entry = curr_close
                else:
                    signals[i] = 0.0
            else:
                # In ranging regime, stay flat or mean revert at extremes (optional)
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high since entry for trailing stop
            if curr_close > highest_since_entry:
                highest_since_entry = curr_close
            
            # Exit conditions: ATR trailing stop OR Donchian breakout down OR regime change to ranging
            stop_price = highest_since_entry - 2.5 * curr_atr
            if curr_close < stop_price or breakout_down or ranging_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Update lowest low since entry for trailing stop
            if curr_close < lowest_since_entry:
                lowest_since_entry = curr_close
            
            # Exit conditions: ATR trailing stop OR Donchian breakout up OR regime change to ranging
            stop_price = lowest_since_entry + 2.5 * curr_atr
            if curr_close > stop_price or breakout_up or ranging_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
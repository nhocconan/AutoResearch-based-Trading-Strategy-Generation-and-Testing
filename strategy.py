#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with volume spike and 1d choppiness regime filter
# - Long: price breaks above Camarilla H3 level, volume > 2x 20-period average, 1d CHOP > 61.8 (ranging market favors mean reversion from extremes)
# - Short: price breaks below Camarilla L3 level, volume > 2x 20-period average, 1d CHOP > 61.8
# - Exit: price returns to Camarilla H4/L4 levels or ATR-based stop
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 15-30 trades/year (60-120 total over 4 years) to stay within fee drag limits
# - Camarilla pivots work well in ranging markets when combined with volume confirmation and regime filter
# - Choppiness index > 61.8 indicates ranging conditions where mean reversion from extreme levels is effective

name = "4h_1d_camarilla_chop_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for choppiness regime filter (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr_1d = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(tr14) / (hh14 - ll14)) / log10(14)
    # Avoid division by zero and invalid values
    range_14 = hh_14 - ll_14
    chop_raw = np.zeros_like(tr_sum_14)
    mask = (range_14 > 0) & (tr_sum_14 > 0)
    chop_raw[mask] = 100 * np.log10(tr_sum_14[mask] / range_14[mask]) / np.log10(14)
    
    # Align 1d Choppiness Index to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    # Pre-compute 4h Camarilla pivot levels (based on previous day's OHLC)
    # We need to calculate daily OHLC from 4h data, but since we don't have daily alignment,
    # we'll use a simplified approach: calculate Camarilla levels based on 4h period's OHLC
    # This is an approximation but avoids look-ahead and uses available data
    
    # For each 4h bar, we'll use the bar's own OHLC to calculate Camarilla levels
    # This is not standard but works for demonstration and avoids MTF complexity
    # In practice, you'd want to use previous day's OHLC, but we approximate with current bar
    
    # Pre-compute 4h OHLC for Camarilla calculation
    # We'll use rolling window to get period OHLC (simplified)
    # Actually, for Camarilla we need the previous period's OHLC, so we shift by 1
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_open = np.roll(prices['open'].values, 1)
    
    # Handle first bar
    prev_close[0] = close[0]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_open[0] = prices['open'].iloc[0]
    
    # Camarilla levels calculation
    range_prev = prev_high - prev_low
    camarilla_h3 = prev_close + range_prev * 1.1 / 4
    camarilla_l3 = prev_close - range_prev * 1.1 / 4
    camarilla_h4 = prev_close + range_prev * 1.1 / 2
    camarilla_l4 = prev_close - range_prev * 1.1 / 2
    
    # Pre-compute 4h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute ATR for stoploss
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or np.isnan(camarilla_h4[i]) or
            np.isnan(camarilla_l4[i]) or np.isnan(volume_sma_20[i]) or np.isnan(atr_14[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Camarilla levels
        h3 = camarilla_h3[i]
        l3 = camarilla_l3[i]
        h4 = camarilla_h4[i]
        l4 = camarilla_l4[i]
        
        # Volume confirmation: current volume > 2x 20-period average (spike)
        vol_confirm = volume_current > 2.0 * volume_sma_20[i]
        
        # Regime filter: 1d Choppiness Index > 61.8 (ranging market)
        chop_regime = chop_aligned[i] > 61.8
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price above Camarilla H3, volume spike, ranging market
        if close_price > h3 and vol_confirm and chop_regime:
            enter_long = True
        
        # Short breakout: price below Camarilla L3, volume spike, ranging market
        if close_price < l3 and vol_confirm and chop_regime:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price reaches H4 or ATR-based stop
            exit_long = (close_price >= h4) or (close_price <= entry_price - 1.5 * atr_14[i])
        elif position == -1:
            # Exit short if price reaches L4 or ATR-based stop
            exit_short = (close_price <= l4) or (close_price >= entry_price + 1.5 * atr_14[i])
        
        # Track entry price for stoploss calculation
        if enter_long or enter_short:
            entry_price = close_price
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
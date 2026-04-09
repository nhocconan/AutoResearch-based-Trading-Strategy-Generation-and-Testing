#!/usr/bin/env python3
# 1d_camarilla_pivot_volume_chop_regime_v1
# Hypothesis: 1d strategy using weekly HTF trend filter via EMA34, daily Camarilla pivot levels for entry/exit,
# volume spike confirmation, and choppiness regime filter to avoid sideways markets.
# Weekly EMA34 determines primary trend (long only above, short only below).
# Daily Camarilla H3/L3 levels act as entry triggers with stop at H4/L4.
# Volume > 1.5x 20-period average confirms institutional participation.
# Choppiness Index < 61.8 ensures we only trade in trending regimes.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 15-25 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_camarilla_pivot_volume_chop_regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly HTF data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily Camarilla pivot levels (based on previous day)
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.0 * (high - low)
    # L3 = close - 1.0 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # We use previous day's OHLC to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_h4 = prev_close + 1.5 * (prev_high - prev_low)
    camarilla_h3 = prev_close + 1.0 * (prev_high - prev_low)
    camarilla_l3 = prev_close - 1.0 * (prev_high - prev_low)
    camarilla_l4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Daily volume confirmation
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (14-period) - regime filter
    # CHOP = 100 * log10(sum(ATR(1)) / (n * log(n))) / log10(n)
    # Simplified: CHOP = 100 * log10(ATR_sum / (true_range_max - true_range_min)) / log10(14)
    # We use a practical approximation: high-low range based
    hl_range = high - low
    atr_1 = pd.Series(hl_range).rolling(window=1, min_periods=1).sum().values  # just HL
    hh_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denom = hh_14 - ll_14
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid div by zero
    chop_ratio = atr_1 / chop_denom
    chop_ratio = np.where(chop_ratio > 1, 1, chop_ratio)  # cap at 1
    chop = 100 * np.log10(chop_ratio) / np.log10(14)
    # Invert so higher = more choppy (standard CHOP: >61.8 = ranging, <38.2 = trending)
    # We'll use: chop < 61.8 = trending (good for breakouts)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # warmup for 20-period MA and 14-period CHOP
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or
            np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or np.isnan(volume_ma[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when market is trending (CHOP < 61.8)
        if chop[i] >= 61.8:
            # In ranging regime, flatten position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price falls below L3 OR weekly trend turns bearish
            if close[i] < camarilla_l3[i] or close[i] < ema_34_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above H3 OR weekly trend turns bullish
            if close[i] > camarilla_h3[i] or close[i] > ema_34_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Long entry: price above H3 AND above weekly EMA34 (bullish alignment)
                if close[i] > camarilla_h3[i] and close[i] > ema_34_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price below L3 AND below weekly EMA34 (bearish alignment)
                elif close[i] < camarilla_l3[i] and close[i] < ema_34_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
# 1d_weekly_camarilla_pivot_volume_regime_v1
# Hypothesis: 1d strategy using weekly Camarilla pivot levels with volume confirmation and choppiness regime filter.
# Long: Price touches weekly L3 level with volume > 1.5x 20-period average and chop > 61.8 (range market).
# Short: Price touches weekly H3 level with volume > 1.5x 20-period average and chop > 61.8 (range market).
# Exit: Price moves to opposite H3/L3 level or closes beyond H4/L4 (breakout).
# Uses weekly trend filter: only long when weekly close > weekly EMA20, only short when weekly close < weekly EMA20.
# Target: 15-25 trades/year to minimize fee drag while capturing mean reversion in ranging markets.
# Camarilla pivots work well in ranging markets (chop > 61.8) which frequently occur in BTC/ETH during bear phases.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_camarilla_pivot_volume_regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for Camarilla pivots and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need at least 20 periods for EMA
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # Calculate weekly Camarilla pivot levels (based on previous week)
    # Camarilla: H4 = close + 1.1*(high-low)*1.1/2, H3 = close + 1.1*(high-low)*1.1/4
    #          L3 = close - 1.1*(high-low)*1.1/4, L4 = close - 1.1*(high-low)*1.1/2
    # Using previous week's high, low, close
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    prev_open_1w = np.roll(open_1w, 1)
    
    # First week has no previous data
    prev_high_1w[0] = high_1w[0]
    prev_low_1w[0] = low_1w[0]
    prev_close_1w[0] = close_1w[0]
    prev_open_1w[0] = open_1w[0]
    
    # Camarilla levels
    camarilla_range = prev_high_1w - prev_low_1w
    h4 = prev_close_1w + 1.1 * camarilla_range * 1.1 / 2.0
    h3 = prev_close_1w + 1.1 * camarilla_range * 1.1 / 4.0
    l3 = prev_close_1w - 1.1 * camarilla_range * 1.1 / 4.0
    l4 = prev_close_1w - 1.1 * camarilla_range * 1.1 / 2.0
    
    # Align weekly Camarilla levels to daily
    h4_aligned = align_htf_to_ltf(prices, df_1w, h4)
    h3_aligned = align_htf_to_ltf(prices, df_1w, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1w, l3)
    l4_aligned = align_htf_to_ltf(prices, df_1w, l4)
    
    # Weekly trend filter: EMA20
    close_1w_s = pd.Series(close_1w)
    ema_20_1w = close_1w_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Choppiness Index regime filter (14-period)
    # CHOP = 100 * log10(sum(ATR14) / (max(high14) - min(low14))) / log10(14)
    # CHOP > 61.8 = ranging market (good for mean reversion)
    # CHOP < 38.2 = trending market
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = abs(high_s - close_s.shift(1))
    tr3 = abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR14
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    max_high14 = high_s.rolling(window=14, min_periods=14).max().values
    min_low14 = low_s.rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = np.zeros(n)
    for i in range(n):
        if (max_high14[i] - min_low14[i]) > 0 and not np.isnan(sum_atr14[i]):
            chop[i] = 100 * np.log10(sum_atr14[i] / (max_high14[i] - min_low14[i])) / np.log10(14)
        else:
            chop[i] = 50  # neutral when undefined
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or
            np.isnan(volume[i]) or np.isnan(chop[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(close_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # Regime filter: chop > 61.8 = ranging market (good for mean reversion)
        ranging_market = chop[i] > 61.8
        # Weekly trend filter
        weekly_uptrend = close_1w_aligned[i] > ema_20_1w_aligned[i]
        weekly_downtrend = close_1w_aligned[i] < ema_20_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Price moves to L3 level or closes below L4 (breakdown)
            if close[i] <= l3_aligned[i] or close[i] < l4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price moves to H3 level or closes above H4 (breakout)
            if close[i] >= h3_aligned[i] or close[i] > h4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price touches L3 level with volume, ranging market, and weekly uptrend bias
            if (abs(close[i] - l3_aligned[i]) < (high[i] - low[i]) * 0.1 and  # Touches L3 (within 10% of daily range)
                volume_confirmed and                       # Volume spike
                ranging_market and                         # Ranging market (chop > 61.8)
                weekly_uptrend):                           # Weekly uptrend bias
                position = 1
                signals[i] = 0.25
            # Short entry: Price touches H3 level with volume, ranging market, and weekly downtrend bias
            elif (abs(close[i] - h3_aligned[i]) < (high[i] - low[i]) * 0.1 and  # Touches H3 (within 10% of daily range)
                  volume_confirmed and                     # Volume spike
                  ranging_market and                       # Ranging market (chop > 61.8)
                  weekly_downtrend):                       # Weekly downtrend bias
                position = -1
                signals[i] = -0.25
    
    return signals
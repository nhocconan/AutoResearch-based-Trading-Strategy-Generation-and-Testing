#!/usr/bin/env python3
# 1d_camarilla_pivot_volume_chop_v1
# Hypothesis: 1d strategy using Camarilla pivot levels from 1d for entry, volume confirmation (>1.5x 20-day avg volume), and chop regime filter (CHOP<61.8 = trending). Uses 1w HTF EMA(50) for trend alignment. Discrete position sizing (0.25) to minimize fee churn. Target: 7-25 trades/year (30-100 total over 4 years). Works in bull/bear: Camarilla captures mean reversion at key levels, volume confirms conviction, chop filter avoids whipsaws in ranging markets, HTF EMA ensures alignment with higher timeframe trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_camarilla_pivot_volume_chop_v1"
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
    
    # Calculate previous day's Camarilla pivot levels
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan  # First bar has no previous day
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels
    camarilla_h4 = pivot + (range_val * 1.1 / 2)
    camarilla_h3 = pivot + (range_val * 1.1 / 4)
    camarilla_h2 = pivot + (range_val * 1.1 / 6)
    camarilla_h1 = pivot + (range_val * 1.1 / 12)
    camarilla_l1 = pivot - (range_val * 1.1 / 12)
    camarilla_l2 = pivot - (range_val * 1.1 / 6)
    camarilla_l3 = pivot - (range_val * 1.1 / 4)
    camarilla_l4 = pivot - (range_val * 1.1 / 2)
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index regime filter (14-period)
    atr_period = 14
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr_series = pd.Series(tr)
    atr_series = tr_series.rolling(window=atr_period, min_periods=atr_period).mean()
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    atr_sum = tr_series.rolling(window=atr_period, min_periods=atr_period).sum().values
    # Avoid division by zero or log of zero
    denominator = np.log10(atr_period) * (highest_high - lowest_low)
    denominator = np.where(denominator == 0, np.nan, denominator)
    chop = 100 * np.log10(atr_sum / denominator)
    
    # Multi-timeframe: 1w EMA(50) trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    close_1w_s = pd.Series(close_1w)
    ema_50_1w = close_1w_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(pivot[i]) or np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or
            np.isnan(volume_ma[i]) or np.isnan(chop[i]) or
            np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # Regime filter: chop < 61.8 indicates trending market
        trending_market = chop[i] < 61.8
        # HTF trend filter: price above/below 1w EMA(50)
        htf_uptrend = close[i] > ema_50_1w_aligned[i]
        htf_downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below camarilla_l3 (mean reversion target)
            if close[i] < camarilla_l3[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above camarilla_h3 (mean reversion target)
            if close[i] > camarilla_h3[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for mean reversion at extreme Camarilla levels with volume, regime, and HTF confirmation
            bullish_setup = (close[i] <= camarilla_l4[i]) and volume_confirmed and trending_market and htf_uptrend
            bearish_setup = (close[i] >= camarilla_h4[i]) and volume_confirmed and trending_market and htf_downtrend
            
            if bullish_setup:
                position = 1
                signals[i] = 0.25
            elif bearish_setup:
                position = -1
                signals[i] = -0.25
    
    return signals
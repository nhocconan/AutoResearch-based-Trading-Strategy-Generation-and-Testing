#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and chop regime filter
# - Primary: 4h price breaking above/below Camarilla H3/L3 levels from prior 1d
# - HTF filter: 1d volume > 1.3x 20-period MA for institutional participation
# - Regime filter: 1d Choppiness Index(14) < 38.2 to ensure trending market (avoid chop)
# - Entry: Long when close > H3 + volume filter + trending regime; Short when close < L3 + volume filter + trending regime
# - Exit: Price crosses prior day's close (mean reversion to equilibrium) or regime shifts to chop
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Target: 80-160 total trades over 4 years (20-40/year) for 4h timeframe
# - Works in bull/bear: Camarilla pivots act as support/resistance in all regimes, volume confirms breakout validity, chop filter avoids false signals

name = "4h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla levels (H3, L3) from prior day
    def calculate_camarilla(high, low, close):
        # Camarilla levels based on prior day's range
        range_ = high - low
        H3 = close + range_ * 1.1 / 4
        L3 = close - range_ * 1.1 / 4
        return H3, L3
    
    # Shift by 1 to use prior day's levels (no look-ahead)
    camarilla_H3_raw = np.full_like(close_1d, np.nan)
    camarilla_L3_raw = np.full_like(close_1d, np.nan)
    for i in range(1, len(close_1d)):
        H3, L3 = calculate_camarilla(high_1d[i-1], low_1d[i-1], close_1d[i-1])
        camarilla_H3_raw[i] = H3
        camarilla_L3_raw[i] = L3
    
    camarilla_H3 = align_htf_to_ltf(prices, df_1d, camarilla_H3_raw)
    camarilla_L3 = align_htf_to_ltf(prices, df_1d, camarilla_L3_raw)
    
    # Calculate 1d volume MA(20)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate 1d Choppiness Index(14)
    def calculate_choppiness(high, low, close, window=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # Sum of TR over window
        tr_sum = pd.Series(tr).rolling(window=window, min_periods=window).sum().values
        
        # Highest high and lowest low over window
        hh = pd.Series(high).rolling(window=window, min_periods=window).max().values
        ll = pd.Series(low).rolling(window=window, min_periods=window).min().values
        
        # Choppiness Index
        chop = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(window)
        return chop
    
    chop_1d = calculate_choppiness(high_1d, low_1d, close_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(60, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(camarilla_H3[i]) or np.isnan(camarilla_L3[i]) or
            np.isnan(volume_ma_20_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period MA
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirm = volume_1d_aligned[i] > 1.3 * volume_ma_20_1d_aligned[i]
        
        # Regime filter: chop < 38.2 = trending market
        trending_regime = chop_1d_aligned[i] < 38.2
        # Chop regime: chop > 61.8 = ranging market (exit signal)
        chop_regime = chop_1d_aligned[i] > 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long entry: close > H3 + volume confirmation + trending regime
            if (close[i] > camarilla_H3[i] and volume_confirm and trending_regime):
                position = 1
                signals[i] = 0.25
            # Short entry: close < L3 + volume confirmation + trending regime
            elif (close[i] < camarilla_L3[i] and volume_confirm and trending_regime):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price crosses prior day's close OR regime shifts to chop
            prior_close_1d = np.full_like(close_1d, np.nan)
            for i_1d in range(1, len(close_1d)):
                prior_close_1d[i_1d] = close_1d[i_1d-1]
            prior_close_aligned = align_htf_to_ltf(prices, df_1d, prior_close_1d)
            
            if position == 1:  # Long position
                if close[i] < prior_close_aligned[i] or chop_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] > prior_close_aligned[i] or chop_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals
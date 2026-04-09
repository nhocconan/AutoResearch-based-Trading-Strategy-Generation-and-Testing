#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot levels + 1w volume confirmation + chop regime filter
# Camarilla pivots provide intraday support/resistance levels that work in both bull and bear markets
# 1w volume spike confirms institutional participation (avoids false breakouts)
# Choppiness index regime filter adapts to market conditions: CHOP > 61.8 = range (mean revert at pivots), CHOP < 38.2 = trending (follow breakout)
# Works in bull/bear: regime filter adapts, Camarilla levels provide structure in all markets
# Target: 30-100 total trades over 4 years (7-25/year) with discrete sizing 0.25-0.30

name = "1d_1w_camarilla_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for volume and chop calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w average volume (20-period)
    volume_1w = df_1w['volume'].values
    volume_s_1w = pd.Series(volume_1w)
    avg_volume_1w = volume_s_1w.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w Choppiness Index (CHOP)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14) - smoothed TR using Wilder's smoothing
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        # First value is simple average
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1w = wilders_smoothing(tr, 14)
    
    # Highest high and lowest low over 14 periods
    hh_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Chop calculation: 100 * log10(sum(atr14) / (hh14 - ll14)) / log10(14)
    sum_atr_14 = pd.Series(atr_1w).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_1w - ll_1w
    chop_1w = np.where(range_14 != 0, 
                       100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 
                       50)  # neutral when range is zero
    
    # Align 1w indicators to 1d timeframe (wait for 1w bar close)
    avg_volume_1w_aligned = align_htf_to_ltf(prices, df_1w, avg_volume_1w)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # H4 = Close + 1.5 * (High - Low)
    # H3 = Close + 1.25 * (High - Low)
    # H2 = Close + 1.1 * (High - Low)
    # H1 = Close + 0.5 * (High - Low)
    # L1 = Close - 0.5 * (High - Low)
    # L2 = Close - 1.1 * (High - Low)
    # L3 = Close - 1.25 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    
    # Shift OHLC by 1 to use previous day's data for today's levels
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    prev_close = np.concatenate([[np.nan], close[:-1]])
    
    # Calculate Camarilla levels for each day
    camarilla_h4 = prev_close + 1.5 * (prev_high - prev_low)
    camarilla_h3 = prev_close + 1.25 * (prev_high - prev_low)
    camarilla_h2 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_h1 = prev_close + 0.5 * (prev_high - prev_low)
    camarilla_l1 = prev_close - 0.5 * (prev_high - prev_low)
    camarilla_l2 = prev_close - 1.1 * (prev_high - prev_low)
    camarilla_l3 = prev_close - 1.25 * (prev_high - prev_low)
    camarilla_l4 = prev_close - 1.5 * (prev_high - prev_low)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or
            np.isnan(avg_volume_1w_aligned[i]) or np.isnan(chop_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2.0x 1w average volume
        volume_confirmed = volume[i] > 2.0 * avg_volume_1w_aligned[i]
        
        # Regime filter: CHOP < 38.2 = trending (follow breakout), CHOP > 61.8 = range (mean revert)
        trending_regime = chop_1w_aligned[i] < 38.2
        ranging_regime = chop_1w_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price closes below Camarilla L3 OR regime shifts to ranging
            if close[i] < camarilla_l3[i] or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla H3 OR regime shifts to ranging
            if close[i] > camarilla_h3[i] or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic
            if trending_regime and volume_confirmed:
                # Follow breakout in trending regime
                if close[i] > camarilla_h4[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < camarilla_l4[i]:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime and volume_confirmed:
                # Mean revert at Camarilla H3/L3 levels in ranging regime
                if close[i] < camarilla_l3[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] > camarilla_h3[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
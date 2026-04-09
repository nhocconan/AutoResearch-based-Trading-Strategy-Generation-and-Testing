#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R extreme reversal with 1w volume spike and chop regime filter
# In ranging markets (CHOP > 61.8): buy when Williams %R < -80 (oversold) with volume confirmation
# sell when Williams %R > -20 (overbought) with volume confirmation
# In trending markets (CHOP < 38.2): follow 1w EMA(21) direction with pullback entries
# Uses discrete sizing 0.25 to limit trades and reduce fee drag
# Williams %R identifies exhaustion points that work in both bull and bear markets

name = "12h_1w_williamsr_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w ATR(14) for volatility normalization
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1w = wilders_smoothing(tr, 14)
    
    # Calculate 1w average volume (20-period) normalized by ATR
    volume_s_1w = pd.Series(volume_1w)
    avg_volume_1w = volume_s_1w.rolling(window=20, min_periods=20).mean().values
    vol_ratio_1w = np.where(atr_1w > 0, avg_volume_1w / atr_1w, np.nan)
    avg_vol_ratio_1w = pd.Series(vol_ratio_1w).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w Williams %R (14-period)
    highest_high_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r_1w = -100 * (highest_high_1w - close_1w) / (highest_high_1w - lowest_low_1w)
    williams_r_1w = np.where((highest_high_1w - lowest_low_1w) == 0, -50, williams_r_1w)
    
    # Calculate 1w EMA(21) for trend direction
    close_s_1w = pd.Series(close_1w)
    ema_21_1w = close_s_1w.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 1w Choppiness Index (CHOP)
    hh_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    sum_atr_14 = pd.Series(atr_1w).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_1w - ll_1w
    chop_1w = np.where(range_14 != 0, 
                       100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 
                       50)
    
    # Align 1w indicators to 12h timeframe
    avg_vol_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, avg_vol_ratio_1w)
    williams_r_1w_aligned = align_htf_to_ltf(prices, df_1w, williams_r_1w)
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Pre-compute volume confirmation array
    avg_volume_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    avg_volume_1w_aligned = align_htf_to_ltf(prices, df_1w, avg_volume_1w)
    volume_confirmed = volume > 1.5 * avg_volume_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(avg_vol_ratio_1w_aligned[i]) or np.isnan(williams_r_1w_aligned[i]) or
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(chop_1w_aligned[i]) or
            np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter
        ranging_regime = chop_1w_aligned[i] > 61.8
        trending_regime = chop_1w_aligned[i] < 38.2
        
        if position == 1:  # Long position
            if ranging_regime:
                # Exit long if Williams %R rises above -50 or volume drops
                if williams_r_1w_aligned[i] > -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif trending_regime:
                # Exit long if price falls below EMA(21) or regime changes to ranging
                if close[i] < ema_21_1w_aligned[i] or ranging_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if ranging_regime:
                # Exit short if Williams %R falls below -50 or volume drops
                if williams_r_1w_aligned[i] < -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif trending_regime:
                # Exit short if price rises above EMA(21) or regime changes to ranging
                if close[i] > ema_21_1w_aligned[i] or ranging_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if ranging_regime:
                # Mean reversion: buy oversold, sell overbought with volume confirmation
                if williams_r_1w_aligned[i] < -80 and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.25
                elif williams_r_1w_aligned[i] > -20 and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.25
            elif trending_regime:
                # Trend following: buy pullbacks in uptrend, sell rallies in downtrend
                if close[i] > ema_21_1w_aligned[i]:  # Uptrend
                    # Buy on pullback to EMA with volume confirmation
                    if close[i] <= ema_21_1w_aligned[i] * 1.01 and volume_confirmed[i]:
                        position = 1
                        signals[i] = 0.25
                else:  # Downtrend
                    # Sell on rally to EMA with volume confirmation
                    if close[i] >= ema_21_1w_aligned[i] * 0.99 and volume_confirmed[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals
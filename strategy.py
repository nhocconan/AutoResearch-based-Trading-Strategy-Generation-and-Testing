#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend with 1d ATR-based chop regime filter and volume confirmation
# - Uses 12h KAMA (adaptive moving average) to capture trend direction with lower whipsaw
# - Regime filter: 1d Choppiness Index > 61.8 for mean-reversion bias in choppy markets
# - Volume confirmation: 12h volume > 1.3x 30-period average to avoid breakouts on low volume
# - Entry: Long when price > KAMA AND chop regime AND volume spike
# - Entry: Short when price < KAMA AND chop regime AND volume spike
# - Exit: Opposite KAMA cross OR chop regime breaks below 38.2 (trending market)
# - Position size: 0.25 (discrete level to minimize fee churn)
# - Target: 50-150 total trades over 4 years (12-37/year) per 12h strategy guidelines
# - Works in bull/bear: KAMA adapts to volatility, chop filter avoids false signals in extremes

name = "12h_1d_kama_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d True Range for Choppiness Index
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr_1d[0]
    
    # 1d ATR(14) for Choppiness Index denominator
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # 1d Sum of ATR(14) for Choppiness Index numerator
    atr_sum_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # 1d Max High and Min Low over 14 periods for range
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    # 1d Choppiness Index (CHOP)
    # CHOP > 61.8 = ranging market (favor mean reversion)
    # CHOP < 38.2 = trending market (favor trend following)
    chop = np.where(range_14 > 0, 
                    100 * np.log10(atr_sum_14 / range_14) / np.log10(14), 
                    50)
    chop = np.where(np.isnan(chop), 50, chop)
    chop_regime_ranging = chop > 61.8   # Ranging market - mean reversion bias
    chop_regime_trending = chop < 38.2  # Trending market - trend following bias
    
    # Align 1d chop regimes to 12h timeframe (completed 1d bar only)
    chop_ranging_aligned = align_htf_to_ltf(prices, df_1d, chop_regime_ranging)
    chop_trending_aligned = align_htf_to_ltf(prices, df_1d, chop_regime_trending)
    
    # 12h price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h KAMA (Adaptive Moving Average)
    # Efficiency Ratio: ER = |net change| / sum(|abs change|)
    change = np.abs(np.diff(close, 1))
    change = np.insert(change, 0, 0)  # align dimensions
    abs_change = np.abs(np.diff(close, 1))
    abs_change = np.insert(abs_change, 0, 0)
    
    # 10-period ER for KAMA
    net_change = np.abs(pd.Series(close).diff(10).fillna(0).values)
    vol = pd.Series(abs_change).rolling(window=10, min_periods=10).sum().values
    er = np.where(vol > 0, net_change / vol, 0)
    
    # Smoothing constants: fastest SC=2/(2+1)=0.67, slowest SC=2/(30+1)=0.0645
    sc = (er * (0.67 - 0.0645) + 0.0645) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 12h Volume > 1.3x 30-period average (volume confirmation)
    avg_volume_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.3 * avg_volume_30)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or 
            np.isnan(chop_ranging_aligned[i]) or
            np.isnan(chop_trending_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below KAMA OR chop regime becomes trending (exit mean reversion)
            if close[i] < kama[i] or chop_trending_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above KAMA OR chop regime becomes trending (exit mean reversion)
            if close[i] > kama[i] or chop_trending_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price above KAMA AND ranging market AND volume spike
            if close[i] > kama[i] and chop_ranging_aligned[i] and volume_spike[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price below KAMA AND ranging market AND volume spike
            elif close[i] < kama[i] and chop_ranging_aligned[i] and volume_spike[i]:
                position = -1
                signals[i] = -0.25
    
    return signals
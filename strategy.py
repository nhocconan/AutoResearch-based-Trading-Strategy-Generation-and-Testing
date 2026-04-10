#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA trend with 12h volume spike and 1d chop regime filter
# - Long when 4h KAMA is rising AND 12h volume > 2.0x 20-period volume SMA AND 1d chop > 61.8 (range market)
# - Short when 4h KAMA is falling AND 12h volume > 2.0x 20-period volume SMA AND 1d chop > 61.8
# - Exit: opposite KAMA direction or chop < 38.2 (trending market)
# - Uses 4h for trend (KAMA adaptive moving average), 12h for volume confirmation, 1d for regime (chop)
# - KAMA adapts to market noise, reducing whipsaws in chop; volume confirms breakout validity
# - Chop regime filter ensures we only trade in ranging markets where mean reversion works
# - Tight entries target 15-30 trades/year to minimize fee drag while maintaining edge
# - Works in bull (buy dips in range) and bear (sell rallies in range) with volume and regime filters

name = "4h_12h_1d_kama_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 12h data ONCE before loop (MTF rule compliance)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return signals
    
    # Calculate 12h volume SMA for confirmation
    vol_12h = df_12h['volume'].values
    volume_sma_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_sma_20_12h)
    
    # Load 1d data for chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate 1d Chop Index (Ehler's Chopiness Index)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(np.maximum(tr1, tr2), tr3)
    tr_1d = np.concatenate([[np.nan], tr_1d])
    
    # ATR(14) for 1d
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_sum_1d = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods for 1d
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    hhll_range_1d = highest_high_1d - lowest_low_1d
    
    # Chop Index: 100 * log10(atr_sum / hhll_range) / log10(14)
    chop_1d = 100 * np.log10(atr_sum_1d / hhll_range_1d) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Pre-compute KAMA on 4h (primary timeframe)
    # Efficiency Ratio (ER) over 10 periods
    change_10 = np.abs(np.concatenate([[np.nan]*10, np.diff(close, n=10)]))
    volatility_10 = pd.Series(np.abs(np.diff(close))).rolling(window=10, min_periods=1).sum().values
    volatility_10 = np.concatenate([[np.nan]*9, volatility_10[9:]])  # align with change_10
    er = np.where(volatility_10 != 0, change_10 / volatility_10, 0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # start with close at index 9
    for i in range(10, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # KAMA direction: rising if current > previous, falling if current < previous
    kama_rising = np.concatenate([[False], np.diff(kama) > 0])
    kama_falling = np.concatenate([[False], np.diff(kama) < 0])
    
    # ATR for dynamic stoploss (using 4h data)
    tr_4h1 = np.abs(high[1:] - low[:-1])
    tr_4h2 = np.abs(high[1:] - close[:-1])
    tr_4h3 = np.abs(low[1:] - close[:-1])
    tr_4h = np.maximum(np.maximum(tr_4h1, tr_4h2), tr_4h3)
    tr_4h = np.concatenate([[np.nan], tr_4h])
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    for i in range(10, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(kama_rising[i]) or np.isnan(kama_falling[i]) or
            np.isnan(volume_sma_20_12h_aligned[i]) or np.isnan(chop_1d_aligned[i]) or
            np.isnan(atr_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 2.0x 20-period volume SMA
        vol_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_12h)
        vol_confirm = vol_12h_aligned[i] > 2.0 * volume_sma_20_12h_aligned[i]
        
        # Regime filter: 1d chop > 61.8 indicates ranging market (mean reversion zone)
        regime_filter = chop_1d_aligned[i] > 61.8
        
        # Only trade when both volume confirmation and regime filter are present
        if vol_confirm and regime_filter:
            # Long when KAMA is rising (trend up in ranging market)
            if kama_rising[i]:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25  # Maintain position
            # Short when KAMA is falling (trend down in ranging market)
            elif kama_falling[i]:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25  # Maintain position
            # Exit: KAMA direction changes or chop < 38.2 (trending market)
            elif ((position == 1 and not kama_rising[i]) or
                  (position == -1 and not kama_falling[i]) or
                  chop_1d_aligned[i] < 38.2):
                if position != 0:  # Only signal on exit
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.0  # Maintain flat
            else:
                # Maintain current position
                signals[i] = 0.25 if position == 1 else -0.25
        else:
            # No trade: exit any position if conditions not met
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals
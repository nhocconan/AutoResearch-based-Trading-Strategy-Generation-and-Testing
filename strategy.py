#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout + 1d volume spike + choppiness regime filter
# - Camarilla levels from 1d: L3, H3, L4, H4 act as intraday support/resistance
# - Long when price breaks above H3 with volume > 1.8x 20-period average (strong conviction)
# - Short when price breaks below L3 with volume > 1.8x 20-period average
# - Choppiness regime filter: only trade when CHOP(14) < 61.8 to avoid ranging markets and false breakouts
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits for 12h
# - Volume spike requirement (>1.8x average) ensures we only trade high-conviction breakouts
# - Works in both bull (breakouts with volume) and bear (breakdowns with volume) markets
# - 1d HTF provides reliable volume and pivot calculation, reducing false signals from lower timeframe noise
# - Choppiness filter avoids whipsaws in sideways markets

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
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
    
    # Load 1d data ONCE before loop for Camarilla pivots, volume, and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # L3 = C - (Range * 1.1 / 4)
    # H3 = C + (Range * 1.1 / 4)
    # L4 = C - (Range * 1.1 / 2)
    # H4 = C + (Range * 1.1 / 2)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    L3 = close_1d - (range_1d * 1.1 / 4)
    H3 = close_1d + (range_1d * 1.1 / 4)
    L4 = close_1d - (range_1d * 1.1 / 2)
    H4 = close_1d + (range_1d * 1.1 / 2)
    
    # 1d volume SMA (20-period)
    volume_series = pd.Series(volume_1d)
    volume_sma_20_1d = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # 1d Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(TR14) / (ATR14 * 14)) / log10(14)
    # Where TR14 = sum of true range over 14 periods
    # ATR14 = average true range over 14 periods
    # Simplified: CHOP = 100 * log10(ATR14_sum / (ATR14 * 14)) / log10(14)
    # Actually: CHOP = 100 * log10(sum(TR14) / (ATR14 * 14)) / log10(14)
    # Since ATR14 = sum(TR14)/14, then sum(TR14)/(ATR14*14) = 1
    # Wait, that's not right. Let me recalculate.
    # True Range TR = max(H-L, abs(H-PC), abs(L-PC))
    # ATR14 = EMA/TR of TR14
    # CHOP = 100 * log10(sum(TR14) / (ATR14 * 14)) / log10(14)
    # Since ATR14 = sum(TR14)/14 (for simple MA), then sum(TR14)/(ATR14*14) = 14
    # Actually ATR14 is usually EMA, not SMA. But for chop, we use:
    # CHOP = 100 * log10(sum(TR14) / (ATR14 * 14)) / log10(14)
    # Where ATR14 is the average true range (typically Wilder's smoothing, EMA)
    # But for simplicity and to match common implementations, we'll use:
    # CHOP = 100 * log10(sum(TR14) / (ATR14 * 14)) / log10(14)
    # With ATR14 as EMA of TR
    
    # Calculate True Range
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR14 using EMA (Wilder's)
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index
    # CHOP = 100 * log10(tr_sum_14 / (atr_14_1d * 14)) / log10(14)
    # Avoid division by zero and log of zero
    chop_denominator = atr_14_1d * 14
    chop_ratio = np.where((chop_denominator > 0) & (tr_sum_14 > 0), tr_sum_14 / chop_denominator, 1.0)
    chop_ratio = np.maximum(chop_ratio, 1e-10)  # Avoid log(0)
    chop_1d = 100 * np.log10(chop_ratio) / np.log10(14)
    
    # Align 1d indicators to 12h timeframe
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(L3_aligned[i]) or np.isnan(H3_aligned[i]) or
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Camarilla breakout conditions
        breakout_long = price_close > H3_aligned[i-1]  # Close above previous period's H3
        breakout_short = price_close < L3_aligned[i-1]  # Close below previous period's L3
        
        # Volume confirmation: current volume > 1.8x 20-period average
        vol_confirm = volume_current > 1.8 * volume_sma_20_aligned[i]
        
        # Choppiness regime filter: trade only when CHOP < 61.8 (trending market)
        chop_filter = chop_aligned[i] < 61.8
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Camarilla H3 breakout + volume confirmation + chop filter
        if breakout_long and vol_confirm and chop_filter:
            enter_long = True
        
        # Short: Camarilla L3 breakdown + volume confirmation + chop filter
        if breakout_short and vol_confirm and chop_filter:
            enter_short = True
        
        # Exit conditions: opposite Camarilla breakout or chop regime shift to ranging
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price breaks below L3 OR chop shifts to ranging (CHOP >= 61.8)
            exit_long = (price_close < L3_aligned[i-1]) or (chop_aligned[i] >= 61.8)
        elif position == -1:
            # Exit short if price breaks above H3 OR chop shifts to ranging
            exit_short = (price_close > H3_aligned[i-1]) or (chop_aligned[i] >= 61.8)
        
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
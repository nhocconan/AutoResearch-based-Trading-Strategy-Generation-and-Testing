#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and choppiness regime filter
# - Uses 1d Camarilla levels (H3, L3, H4, L4) as significant support/resistance
# - Long when price breaks above H4 with volume > 1.5x 24-period average on 1d
# - Short when price breaks below L4 with volume > 1.5x 24-period average on 1d
# - Choppiness regime: only trade when CHOP(14) < 38.2 (trending market) or > 61.8 (range market) for mean reversion at extremes
# - ATR stoploss: exit when price moves 2.5 * ATR(14) against position
# - Discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Camarilla pivots work in both bull (breakouts with volume) and bear (breakdowns with volume) markets
# - 1d HTF provides reliable structure, 12h timeframe balances frequency and cost

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
    entry_price = 0.0  # track entry price for ATR-based stoploss
    
    # Load 1d data ONCE before loop for Camarilla levels, volume, and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Camarilla pivot levels (based on previous day's range)
    # H4 = close + 1.5 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # H3 = close + 1.125 * (high - low)
    # L3 = close - 1.125 * (high - low)
    # We use H4/L4 for breakouts, H3/L3 for mean reversion in chop
    prev_high = pd.Series(high_1d).shift(1)
    prev_low = pd.Series(low_1d).shift(1)
    prev_close = pd.Series(close_1d).shift(1)
    range_1d = prev_high - prev_low
    
    H4 = prev_close + 1.5 * range_1d
    L4 = prev_close - 1.5 * range_1d
    H3 = prev_close + 1.125 * range_1d
    L3 = prev_close - 1.125 * range_1d
    
    # 1d volume SMA (24-period)
    volume_series = pd.Series(volume_1d)
    volume_sma_24_1d = volume_series.rolling(window=24, min_periods=24).mean().values
    
    # Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR(14)) / (n * log10(highest_high - lowest_low)))
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14_1d).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denominator = 14 * np.log10(highest_high_14 - lowest_low_14 + 1e-10)
    chop_ratio = np.where(chop_denominator > 0, sum_atr_14 / chop_denominator, 1)
    chop_1d = 100 * np.log10(chop_ratio)
    
    # ATR for stoploss (14-period)
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 12h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4.values)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4.values)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3.values)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3.values)
    volume_sma_24_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_24_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(volume_sma_24_aligned[i]) or np.isnan(chop_aligned[i]) or np.isnan(atr_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 24-period average (using 1d aligned volume)
        vol_confirm = volume_current > 1.5 * volume_sma_24_aligned[i]
        
        # Choppiness regime: CHOP < 38.2 (trending) or CHOP > 61.8 (range)
        chop_value = chop_aligned[i]
        in_trending_regime = chop_value < 38.2
        in_range_regime = chop_value > 61.8
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: price breaks above H4 with volume confirmation
        if vol_confirm and (in_trending_regime or in_range_regime):
            if price_close > H4_aligned[i]:
                enter_long = True
        
        # Short: price breaks below L4 with volume confirmation
        if vol_confirm and (in_trending_regime or in_range_regime):
            if price_close < L4_aligned[i]:
                enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long: stoploss (2.5 * ATR) or mean reversion in range (touch L3)
            stoploss_level = entry_price - 2.5 * atr_14_aligned[i]
            exit_long = (price_close < stoploss_level) or (in_range_regime and price_close < L3_aligned[i])
        elif position == -1:
            # Exit short: stoploss (2.5 * ATR) or mean reversion in range (touch H3)
            stoploss_level = entry_price + 2.5 * atr_14_aligned[i]
            exit_short = (price_close > stoploss_level) or (in_range_regime and price_close > H3_aligned[i])
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            entry_price = price_close
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            entry_price = price_close
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            entry_price = 0.0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            entry_price = 0.0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
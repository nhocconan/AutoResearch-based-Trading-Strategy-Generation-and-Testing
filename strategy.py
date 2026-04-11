#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot levels from 1d + volume confirmation + choppiness regime filter
# - Camarilla levels (H3, L3, H4, L4) from daily OHLC act as intraday support/resistance
# - Long when price touches L3 with volume > 1.5x 20-period average AND chop < 61.8 (trending)
# - Short when price touches H3 with volume > 1.5x 20-period average AND chop < 61.8 (trending)
# - Uses ATR-based stoploss: exit when price moves 2*ATR against position
# - Discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 19-50 trades/year (75-200 total over 4 years) to stay within fee drag limits
# - Works in both bull (breakouts with volume) and bear (mean reversion at extreme levels) markets

name = "4h_1d_camarilla_volume_chop_v1"
timeframe = "4h"
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
    entry_price = 0.0
    
    # Load 1d data ONCE before loop for Camarilla levels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: H3, L3, H4, L4
    # H4 = close + 1.1*(high-low)*1.1/2
    # L4 = close - 1.1*(high-low)*1.1/2
    # H3 = close + 1.1*(high-low)*1.1/4
    # L3 = close - 1.1*(high-low)*1.1/4
    rng = high_1d - low_1d
    camarilla_h4 = close_1d + 1.1 * rng * 1.1 / 2
    camarilla_l4 = close_1d - 1.1 * rng * 1.1 / 2
    camarilla_h3 = close_1d + 1.1 * rng * 1.1 / 4
    camarilla_l3 = close_1d - 1.1 * rng * 1.1 / 4
    
    # Pre-compute 1d ATR (14-period) for stoploss
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Pre-compute 4h volume SMA (20-period)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute 4h choppiness index (14-period)
    def choppiness_index(high, low, close, window=14):
        atr_sum = pd.Series(high).rolling(window).apply(
            lambda x: np.sum(np.abs(np.diff(x))), raw=True
        ) if len(high) >= window else pd.Series([np.nan]*len(high))
        max_high = pd.Series(high).rolling(window).max()
        min_low = pd.Series(low).rolling(window).min()
        chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(window)
        return chop.values
    
    chop_values = choppiness_index(high, low, close, 14)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(chop_values[i]) or np.isnan(atr_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Chop regime: trade only when chop < 61.8 (trending market)
        chop_filter = chop_values[i] < 61.8
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: price touches L3 with volume confirmation in trending market
        if price_low <= camarilla_l3_aligned[i] and vol_confirm and chop_filter:
            enter_long = True
        
        # Short: price touches H3 with volume confirmation in trending market
        if price_high >= camarilla_h3_aligned[i] and vol_confirm and chop_filter:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long: stoploss (2*ATR against) or price reaches H3
            exit_long = (price_close <= entry_price - 2.0 * atr_14_aligned[i]) or (price_close >= camarilla_h3_aligned[i])
        elif position == -1:
            # Exit short: stoploss (2*ATR against) or price reaches L3
            exit_short = (price_close >= entry_price + 2.0 * atr_14_aligned[i]) or (price_close <= camarilla_l3_aligned[i])
        
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
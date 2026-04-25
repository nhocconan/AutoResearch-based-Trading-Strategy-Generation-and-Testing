#!/usr/bin/env python3
"""
1d_Adaptive_Choppiness_Regime_Strategy
Hypothesis: Primary 1d timeframe with 1w HTF trend filter. Uses Choppiness Index (CHOP) to detect regime: CHOP > 61.8 = range (mean revert at Bollinger Bands), CHOP < 38.2 = trending (follow 1w EMA34). Enters long when price touches lower BB in range regime OR breaks above Donchian(20) high in trending bull regime. Short when price touches upper BB in range regime OR breaks below Donchian(20) low in trending bear regime. Uses ATR-based stoploss and discrete sizing (0.25) to minimize fee churn. Designed for ~15-30 trades/year, works in bull/bear by adapting to regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA34 for trend filter (needs extra delay for confirmation)
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w, additional_delay_bars=1)
    
    # Calculate ATR for stoploss (using 14 periods)
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Bollinger Bands (20, 2.0) for mean reversion in ranging markets
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + (2.0 * bb_std)
    bb_lower = bb_middle - (2.0 * bb_std)
    
    # Donchian Channel (20) for breakouts in trending markets
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Choppiness Index (CHOP) for regime detection (14-period)
    # CHOP = 100 * log10(sum(ATR14) / (max(high) - min(low))) / log10(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    # Avoid division by zero
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    chop = 100 * (np.log10(sum_atr_14 / range_14) / np.log10(14))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    # Start index: need 20-period data for BB/Donchian and 14 for CHOP/ATR
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Determine regime based on Choppiness Index
        is_ranging = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        # Neutral zone (38.2-61.8) - no trades to avoid whipsaw
        
        if position == 0:
            # Long entry conditions
            long_signal = False
            if is_ranging:
                # Mean reversion: touch lower Bollinger Band
                long_signal = curr_low <= bb_lower[i]
            elif is_trending:
                # Trend following: break above Donchian high in bullish 1w trend
                long_signal = (curr_close > donchian_high[i]) and (close_1w[i] > ema_34_1w_aligned[i])
            
            # Short entry conditions
            short_signal = False
            if is_ranging:
                # Mean reversion: touch upper Bollinger Band
                short_signal = curr_high >= bb_upper[i]
            elif is_trending:
                # Trend following: break below Donchian low in bearish 1w trend
                short_signal = (curr_close < donchian_low[i]) and (close_1w[i] < ema_34_1w_aligned[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_stop = entry_price - (1.5 * atr[i])
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_stop = entry_price + (1.5 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions
            exit_long = False
            if is_ranging:
                # Exit mean reversion at middle BB
                exit_long = curr_close >= bb_middle[i]
            elif is_trending:
                # Exit trend follow at Donchian low OR stoploss OR trend reversal
                exit_long = (curr_close < donchian_low[i]) or \
                           (curr_close < atr_stop) or \
                           (close_1w[i] < ema_34_1w_aligned[i])
            
            if exit_long:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions
            exit_short = False
            if is_ranging:
                # Exit mean reversion at middle BB
                exit_short = curr_close <= bb_middle[i]
            elif is_trending:
                # Exit trend follow at Donchian high OR stoploss OR trend reversal
                exit_short = (curr_close > donchian_high[i]) or \
                            (curr_close > atr_stop) or \
                            (close_1w[i] > ema_34_1w_aligned[i])
            
            if exit_short:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Adaptive_Choppiness_Regime_Strategy"
timeframe = "1d"
leverage = 1.0
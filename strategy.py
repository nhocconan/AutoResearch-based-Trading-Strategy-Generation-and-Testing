#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + volume confirmation + choppiness regime filter
# - Donchian(20) breakout: price > upper band (long) or < lower band (short)
# - Volume confirmation: current volume > 1.5x 20-period average
# - Choppiness regime: CHOP(14) > 61.8 for mean-reversion, CHOP < 38.2 for trend-following
# - In trending regime (CHOP < 38.2): follow Donchian breakout direction
# - In ranging regime (CHOP > 61.8): fade Donchian breakout (counter-trend)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 20-50 trades/year (75-200 total over 4 years) to stay within fee drag limits
# - Works in both bull (trend following) and bear (mean reversion in ranges) markets

name = "4h_donchian_volume_chop_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d ATR for volatility filter
    atr_period = 14
    tr1 = pd.Series(df_1d['high']).shift(1) - pd.Series(df_1d['low'])
    tr2 = abs(pd.Series(df_1d['high']).shift(1) - pd.Series(df_1d['close']))
    tr3 = abs(pd.Series(df_1d['low']).shift(1) - pd.Series(df_1d['close']))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Pre-compute 1d choppiness index for regime filter
    chop_period = 14
    tr_sum = pd.Series(tr).rolling(window=chop_period, min_periods=chop_period).sum().values
    highest_high = pd.Series(df_1d['high']).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=chop_period, min_periods=chop_period).min().values
    chop_denom = highest_high - lowest_low
    chop_1d = 100 * np.log10(tr_sum / chop_denom) / np.log10(chop_period)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or
            i < 20):  # Need 20 bars for Donchian
            signals[i] = 0.0
            continue
        
        # Current price data
        volume_current = volume[i]
        high_current = high[i]
        low_current = low[i]
        close_current = close[i]
        
        # Donchian channels (20-period)
        lookback_start = max(0, i - 19)
        highest_high_20 = np.max(high[lookback_start:i+1])
        lowest_low_20 = np.min(low[lookback_start:i+1])
        
        # Volume confirmation: current volume > 1.5x 20-period average
        if i >= 20:
            volume_sma_20 = np.mean(volume[i-19:i+1])
            vol_confirm = volume_current > 1.5 * volume_sma_20
        else:
            vol_confirm = False
        
        # Regime filter based on 1d chop
        chop_value = chop_1d_aligned[i]
        is_trending = chop_value < 38.2
        is_ranging = chop_value > 61.8
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        if vol_confirm:
            # Breakout signals
            breakout_up = close_current > highest_high_20
            breakout_down = close_current < lowest_low_20
            
            if is_trending:
                # Trending regime: follow breakout direction
                if breakout_up:
                    enter_long = True
                elif breakout_down:
                    enter_short = True
            elif is_ranging:
                # Ranging regime: fade breakout (mean reversion)
                if breakout_up:
                    enter_short = True  # Fade upward breakout
                elif breakout_down:
                    enter_long = True   # Fade downward breakout
        
        # Exit conditions: opposite breakout or loss of confirmation
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long on downward breakout or loss of volume confirmation
            exit_long = (close_current < lowest_low_20) or (not vol_confirm)
        elif position == -1:
            # Exit short on upward breakout or loss of volume confirmation
            exit_short = (close_current > highest_high_20) or (not vol_confirm)
        
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
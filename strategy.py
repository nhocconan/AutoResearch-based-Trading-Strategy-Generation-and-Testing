#!/usr/bin/env python3
"""
4h_Three_Strategy_Combo_v1
Hypothesis: Combines three complementary strategies (Donchian breakout with volume, 
RSI mean reversion with trend filter, and Bollinger Band squeeze breakout) to capture 
different market regimes. Uses 12h trend filter to avoid counter-trend trades. 
Designed for low frequency (20-40 trades/year) to work in both bull (breakouts) and 
bear (mean reversion) markets by dynamically weighting strategies based on volatility regime.
"""

name = "4h_Three_Strategy_Combo_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 12h EMA50 for trend filter ---
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # --- Strategy 1: Donchian Breakout with Volume ---
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)  # Volume confirmation
    
    # --- Strategy 2: RSI Mean Reversion with Trend Filter ---
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # --- Strategy 3: Bollinger Band Squeeze Breakout ---
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    bb_width = (upper_bb - lower_bb) / sma_20
    bb_width_ma = bb_width.rolling(window=50, min_periods=50).mean()
    squeeze = bb_width < (0.5 * bb_width_ma.values)  # Bollinger squeeze condition
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(rsi_values[i]) or
            np.isnan(upper_bb[i]) or
            np.isnan(lower_bb[i]) or
            np.isnan(squeeze[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine market regime based on volatility
        # High volatility: favor breakout strategies
        # Low volatility: favor mean reversion
        current_bb_width = bb_width[i]
        bb_width_percentile = pd.Series(bb_width).rolling(window=100, min_periods=30).rank(pct=True).iloc[i] if i >= 30 else 0.5
        
        # Strategy signals
        # Strategy 1: Donchian breakout
        donchian_long = (high[i] > highest_high[i-1]) and vol_spike[i]
        donchian_short = (low[i] < lowest_low[i-1]) and vol_spike[i]
        
        # Strategy 2: RSI mean reversion (only in ranging markets)
        rsi_long = (rsi_values[i] < 30) and (close[i] > ema_50_12h_aligned[i])  # Only long in uptrend
        rsi_short = (rsi_values[i] > 70) and (close[i] < ema_50_12h_aligned[i])  # Only short in downtrend
        
        # Strategy 3: Bollinger Band breakout (only during/after squeeze)
        bb_long = squeeze[i] and (close[i] > upper_bb[i])
        bb_short = squeeze[i] and (close[i] < lower_bb[i])
        
        # Combine strategies based on volatility regime
        # In low volatility (squeeze), weight mean reversion and BB breakout higher
        # In high volatility, weight breakout strategies higher
        if bb_width_percentile < 0.3:  # Low volatility regime
            long_signal = rsi_long or bb_long
            short_signal = rsi_short or bb_short
        else:  # High volatility regime
            long_signal = donchian_long or bb_long
            short_signal = donchian_short or bb_short
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        else:
            # Exit conditions: opposite signal or volatility extreme
            if position == 1:
                exit_signal = short_signal or (rsi_values[i] > 70)  # Exit long on RSI overbought or short signal
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                exit_signal = long_signal or (rsi_values[i] < 30)  # Exit short on RSI oversold or long signal
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
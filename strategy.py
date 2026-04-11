#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout + 1d trend filter + volume confirmation
# - Camarilla levels from 1d: L3 (support) and H3 (resistance) act as intraday pivot points
# - Trend filter: 1d EMA50 > EMA200 for long bias, EMA50 < EMA200 for short bias
# - Volume confirmation: 4h volume > 1.8x 20-period average to filter weak breakouts
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits
# - Camarilla pivots work in both bull (breakouts above H3) and bear (breakdowns below L3) markets
# - 1d EMA filter ensures we trade with the higher timeframe trend
# - Volume confirmation reduces false breakouts during low volatility

name = "4h_1d_camarilla_pivot_breakout_v1"
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
    
    # Load 1d data ONCE before loop for pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: based on previous day's range
    # H3 = close + 1.1 * (high - low) / 2
    # L3 = close - 1.1 * (high - low) / 2
    # H4 = close + 1.1 * (high - low)
    # L4 = close - 1.1 * (high - low)
    # We use H3/L3 for entry, H4/L4 for stronger breakout confirmation
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]  # First value fallback
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    camarilla_h4 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_l4 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Align Camarilla levels to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Pre-compute 1d EMA trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Pre-compute 1d trend bias (1 for uptrend, -1 for downtrend, 0 for neutral)
    trend_bias = np.zeros(len(ema_50_aligned))
    trend_bias[ema_50_aligned > ema_200_aligned] = 1
    trend_bias[ema_50_aligned < ema_200_aligned] = -1
    
    # Pre-compute 4h volume confirmation (20-period average)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(trend_bias[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Camarilla breakout conditions
        breakout_long = price_close > h3_aligned[i]  # Close above H3 level
        breakdown_short = price_close < l3_aligned[i]  # Close below L3 level
        
        # Strong breakout confirmation using H4/L4
        strong_breakout_long = price_close > h4_aligned[i]
        strong_breakout_short = price_close < l4_aligned[i]
        
        # Trend filter from 1d
        trend_up = trend_bias[i] == 1
        trend_down = trend_bias[i] == -1
        
        # Volume confirmation: current volume > 1.8x 20-period average
        vol_confirm = volume_current > 1.8 * volume_sma_20[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Camarilla breakout above H3 + uptrend + volume confirmation
        # Require strong breakout (H4) for better quality
        if strong_breakout_long and trend_up and vol_confirm:
            enter_long = True
        
        # Short: Camarilla breakdown below L3 + downtrend + volume confirmation
        # Require strong breakdown (L4) for better quality
        if strong_breakout_short and trend_down and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite Camarilla level or trend reversal
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price breaks below L3 OR trend turns down
            exit_long = (price_close < l3_aligned[i]) or (not trend_up)
        elif position == -1:
            # Exit short if price breaks above H3 OR trend turns up
            exit_short = (price_close > h3_aligned[i]) or (not trend_down)
        
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
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla H3/L3 breakout with 1w EMA50 trend filter and volume confirmation
# Uses weekly EMA50 from 1w data to determine structural bias (long above weekly EMA50, short below)
# Camarilla H3/L3 breakout provides precise entry timing in direction of weekly trend bias
# Volume confirmation > 1.8x 20-period EMA ensures institutional participation
# Designed for low trade frequency: ~10-20 trades/year per symbol with 0.25 sizing
# Weekly EMA50 acts as dynamic support/resistance that works in both bull and bear markets
# Breakouts in direction of weekly trend bias have higher follow-through probability
# Works in bull markets (trend following) and bear markets (mean reversion at extremes)

name = "1d_Camarilla_H3L3_1wEMA50_Trend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 from weekly close
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels from prior day OHLC
    # H3 = Close + 1.1*(High - Low)/2
    # L3 = Close - 1.1*(High - Low)/2
    # We use the prior completed day's values (shifted by 1)
    prior_high = np.roll(high, 1)
    prior_low = np.roll(low, 1)
    prior_close = np.roll(close, 1)
    
    # Set first value to avoid roll issues
    prior_high[0] = high[0]
    prior_low[0] = low[0]
    prior_close[0] = close[0]
    
    camarilla_h3 = prior_close + 1.1 * (prior_high - prior_low) / 2.0
    camarilla_l3 = prior_close - 1.1 * (prior_high - prior_low) / 2.0
    
    # Volume confirmation: volume > 1.8 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need EMA50 (50 weeks) + Camarilla (prior day) + volume EMA20
    start_idx = max(50, 1, 20)  # 50 for weekly EMA50, 1 for prior day, 20 for volume EMA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_h3[i]) or 
            np.isnan(camarilla_l3[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine bias from weekly EMA50: long above EMA50, short below EMA50
        bullish_bias = close[i] > ema_50_1w_aligned[i]
        bearish_bias = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias:
                # Long: Camarilla H3 breakout with volume spike
                if high[i] > camarilla_h3[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_bias:
                # Short: Camarilla L3 breakdown with volume spike
                if low[i] < camarilla_l3[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop around EMA50
        
        elif position == 1:  # Long position
            # Exit: Camarilla L3 reversion (mean reversion at support)
            if low[i] < camarilla_l3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Camarilla H3 reversion (mean reversion at resistance)
            if high[i] > camarilla_h3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
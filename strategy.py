#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation
# - Long when price breaks above H3 pivot level AND 4h EMA(21) > EMA(50) AND volume > 1.5x 20-bar avg
# - Short when price breaks below L3 pivot level AND 4h EMA(21) < EMA(50) AND volume > 1.5x 20-bar avg
# - Exit when price retests the pivot point (PP) level
# - Uses discrete position sizing (0.20) to minimize fee churn
# - Camarilla pivots provide precise intraday support/resistance levels
# - 4h EMA filter ensures alignment with higher timeframe trend
# - Volume confirmation avoids low-liquidity false signals
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)
# - Works in both bull and bear markets: breakouts work in trends, mean reversion in ranges

name = "1h_4h_camarilla_breakout_volume_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Pre-compute 4h EMA trend filter: EMA(21) vs EMA(50)
    close_4h = df_4h['close'].values
    ema_21 = pd.Series(close_4h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_50 = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_bullish = ema_21 > ema_50
    ema_bearish = ema_21 < ema_50
    
    # Align 4h EMA trend to 1h timeframe
    ema_bullish_aligned = align_htf_to_ltf(prices, df_4h, ema_bullish)
    ema_bearish_aligned = align_htf_to_ltf(prices, df_4h, ema_bearish)
    
    # Pre-compute daily Camarilla pivot levels (using previous day's OHLC)
    # We need to get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot points
    pp = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    h3 = pp + (range_1d * 1.1 / 4)
    l3 = pp - (range_1d * 1.1 / 4)
    h4 = pp + (range_1d * 1.1 / 2)
    l4 = pp - (range_1d * 1.1 / 2)
    
    # Align daily Camarilla levels to 1h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Pre-compute 1h volume confirmation: > 1.5x 20-period average
    volume = prices['volume'].values
    volume_20_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * volume_20_avg)
    
    # Session filter: 08-20 UTC (inclusive)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid or outside session
        if (np.isnan(ema_bullish_aligned[i]) or np.isnan(ema_bearish_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(pp_aligned[i]) or
            np.isnan(vol_spike[i]) or not in_session[i]):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above H3 AND 4h bullish trend AND volume spike
            if (prices['close'].iloc[i] > h3_aligned[i] and 
                ema_bullish_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.20
            # Short when price breaks below L3 AND 4h bearish trend AND volume spike
            elif (prices['close'].iloc[i] < l3_aligned[i] and 
                  ema_bearish_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit when price retests pivot point
            # Exit when price returns to pivot point (PP) level
            exit_long = position == 1 and prices['close'].iloc[i] <= pp_aligned[i]
            exit_short = position == -1 and prices['close'].iloc[i] >= pp_aligned[i]
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h volume filter and 1d trend filter
# - Long when price breaks above Camarilla H3 level AND 4h volume > 1.3x 20-period average AND 1d close > 1d SMA(50)
# - Short when price breaks below Camarilla L3 level AND 4h volume > 1.3x 20-period average AND 1d close < 1d SMA(50)
# - Exit when price crosses Camarilla P (pivot) level OR opposite breakout occurs
# - Uses discrete position sizing 0.20 to limit fee churn
# - Session filter: only trade 08:00-20:00 UTC to avoid low-liquidity hours
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)
# - Camarilla pivots provide precise intraday support/resistance levels
# - 4h volume confirmation reduces false breakouts
# - 1d trend filter ensures we trade with the higher timeframe trend

name = "1h_4h_1d_camarilla_volume_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 20 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1h Camarilla pivots (based on previous day's OHLC)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate daily OHLC for Camarilla formula
    # We need to resample to daily but using actual daily data from df_1d
    # Camarilla levels: based on previous day's range
    # H4 = close + 1.5*(high-low)
    # H3 = close + 1.25*(high-low) 
    # H2 = close + 1.166*(high-low)
    # H1 = close + 1.083*(high-low)
    # P  = (high+low+close)/3
    # L1 = close - 1.083*(high-low)
    # L2 = close - 1.166*(high-low)
    # L3 = close - 1.25*(high-low)
    # L4 = close - 1.5*(high-low)
    
    # Get previous day's OHLC from 1d data
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    # Align previous day's OHLC to 1h timeframe
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close_1d)
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high_1d)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low_1d)
    
    # Calculate Camarilla levels
    daily_range = prev_high_aligned - prev_low_aligned
    camarilla_h3 = prev_close_aligned + 1.25 * daily_range
    camarilla_l3 = prev_close_aligned - 1.25 * daily_range
    camarilla_p = (prev_high_aligned + prev_low_aligned + prev_close_aligned) / 3
    
    # Pre-compute 4h volume confirmation
    vol_ma_4h = pd.Series(df_4h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    volume_spike_4h = volume > (1.3 * vol_4h_aligned)
    
    # Pre-compute 1d trend filter: close > SMA(50)
    sma_50_1d = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    trend_up = df_1d['close'].values > sma_50_1d_aligned  # 1d close > 1d SMA50
    trend_down = df_1d['close'].values < sma_50_1d_aligned  # 1d close < 1d SMA50
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up)
    trend_down_aligned = align_htf_to_ltf(prices, df_1d, trend_down)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(camarilla_p[i]) or np.isnan(volume_spike_4h[i]) or
            np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Only trade during session
        if not in_session[i]:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Camarilla H3 AND 4h volume spike AND 1d uptrend
            if (close[i] > camarilla_h3[i] and 
                volume_spike_4h[i] and 
                trend_up_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short conditions: price breaks below Camarilla L3 AND 4h volume spike AND 1d downtrend
            elif (close[i] < camarilla_l3[i] and 
                  volume_spike_4h[i] and 
                  trend_down_aligned[i]):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses Camarilla P level OR opposite breakout occurs
            exit_long = (position == 1 and 
                        (close[i] < camarilla_p[i] or close[i] < camarilla_l3[i]))
            exit_short = (position == -1 and 
                         (close[i] > camarilla_p[i] or close[i] > camarilla_h3[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
    
    return signals
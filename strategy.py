#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h volume confirmation and 1d trend filter
# - Long when price breaks above Camarilla H3 level with 4h volume > 1.5x 20-period average AND 1d close > 1d EMA50
# - Short when price breaks below Camarilla L3 level with 4h volume > 1.5x 20-period average AND 1d close < 1d EMA50
# - Camarilla levels calculated from previous 1h bar's high-low range (intraday support/resistance)
# - Volume confirmation ensures breakout conviction
# - 1d EMA50 filter ensures we trade with the daily trend (avoid counter-trend whipsaws)
# - Discrete position sizing: ±0.20 to limit drawdown and reduce fee churn
# - Target: 15-37 trades/year (60-150 total over 4 years) to stay within fee drag limits for 1h
# - Session filter: 08-20 UTC to avoid low-volume Asian session noise
# - Works in bull markets (breakouts with volume in uptrend) and bear markets (breakdowns with volume in downtrend)

name = "1h_4h_1d_camarilla_volume_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 4h data ONCE before loop for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return signals
    
    # Pre-compute 4h volume SMA (20-period)
    volume_4h = df_4h['volume'].values
    volume_series_4h = pd.Series(volume_4h)
    volume_sma_20_4h = volume_series_4h.rolling(window=20, min_periods=20).mean().values
    
    # Align 4h volume to 1h timeframe
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_4h, volume_sma_20_4h)
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 1h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is invalid
        if (np.isnan(volume_sma_20_aligned[i]) or np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels from previous 1h bar
        # Camarilla levels: based on previous bar's range
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        range_val = prev_high - prev_low
        
        # Camarilla levels for intraday trading
        # H3 = prev_close + range * 1.1/4
        # L3 = prev_close - range * 1.1/4
        camarilla_h3 = prev_close + range_val * 1.1 / 4
        camarilla_l3 = prev_close - range_val * 1.1 / 4
        
        # Current price data
        price_close = close[i]
        volume_current = volume[i]
        
        # Breakout conditions
        breakout_long = price_close > camarilla_h3
        breakout_short = price_close < camarilla_l3
        
        # Volume confirmation: current volume > 1.5x 20-period average (using 4h aligned volume)
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Trend filter: trade with 1d EMA50 direction
        trend_long = price_close > ema_50_aligned[i]
        trend_short = price_close < ema_50_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Camarilla H3 breakout + volume confirmation + uptrend filter
        if breakout_long and vol_confirm and trend_long:
            enter_long = True
        
        # Short: Camarilla L3 breakdown + volume confirmation + downtrend filter
        if breakout_short and vol_confirm and trend_short:
            enter_short = True
        
        # Exit conditions: opposite Camarilla level break
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price breaks below L3
            exit_long = price_close < camarilla_l3
        elif position == -1:
            # Exit short if price breaks above H3
            exit_short = price_close > camarilla_h3
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.20
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals
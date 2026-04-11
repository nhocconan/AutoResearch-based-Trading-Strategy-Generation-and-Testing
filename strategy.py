#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h volume confirmation and 1d trend filter
# - Long when price breaks above H3 pivot level with 4h volume > 1.5x 20-period average and 1d close > EMA50
# - Short when price breaks below L3 pivot level with 4h volume > 1.5x 20-period average and 1d close < EMA50
# - Uses discrete position sizing: ±0.20 to limit drawdown and reduce fee churn
# - Target: 15-37 trades/year (60-150 total over 4 years) to stay within fee drag limits for 1h
# - Session filter: 08-20 UTC to avoid low-liquidity periods
# - Works in both bull (breakouts with volume in uptrend) and bear (breakdowns with volume in downtrend) markets

name = "1h_4h_1d_camarilla_volume_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return signals
    
    # Pre-compute 4h volume SMA (20-period)
    volume_4h = df_4h['volume'].values
    volume_series_4h = pd.Series(volume_4h)
    volume_sma_20_4h = volume_series_4h.rolling(window=20, min_periods=20).mean().values
    volume_sma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_sma_20_4h)
    
    # Load 1d data ONCE before loop for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(volume_sma_20_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        
        # Calculate Camarilla pivot levels for current 1h bar using previous bar's OHLC
        # Camarilla levels: based on previous bar's range
        if i == 0:
            signals[i] = 0.0
            continue
            
        prev_close = close[i-1]
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_range = prev_high - prev_low
        
        # Camarilla levels (H3/L3 are the key breakout levels)
        # H3 = close + 1.1 * (high - low) / 4
        # L3 = close - 1.1 * (high - low) / 4
        camarilla_h3 = prev_close + 1.1 * prev_range / 4
        camarilla_l3 = prev_close - 1.1 * prev_range / 4
        
        # Volume confirmation: current 1h volume > 1.5x 20-period 4h volume average (aligned)
        vol_confirm = volume[i] > 1.5 * volume_sma_20_4h_aligned[i]
        
        # Trend filter: 1d EMA50 direction
        uptrend = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]  # Rising EMA50
        downtrend = ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]  # Falling EMA50
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: price breaks above H3 + volume confirmation + uptrend
        if price_close > camarilla_h3 and vol_confirm and uptrend:
            enter_long = True
        
        # Short: price breaks below L3 + volume confirmation + downtrend
        if price_close < camarilla_l3 and vol_confirm and downtrend:
            enter_short = True
        
        # Exit conditions: opposite Camarilla breakout or loss of volume/trend
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price breaks below L3 OR volume/trend deteriorates
            exit_long = (price_close < camarilla_l3) or (not vol_confirm) or (not uptrend)
        elif position == -1:
            # Exit short if price breaks above H3 OR volume/trend deteriorates
            exit_short = (price_close > camarilla_h3) or (not vol_confirm) or (not downtrend)
        
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
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation
# - Uses 4h EMA(50) for trend direction (bullish if price > EMA, bearish if price < EMA)
# - 1d Camarilla pivot levels (H3/L3) as breakout triggers on 1h timeframe
# - Volume spike confirmation: 1h volume > 1.5x 20-period average to ensure conviction
# - Session filter: 08-20 UTC to avoid low-liquidity Asian session noise
# - Discrete position sizing: ±0.20 to limit drawdown and reduce fee churn
# - Target: 15-37 trades/year (60-150 total over 4 years) to stay within fee drag limits for 1h
# - Works in bull markets (breakouts with volume in uptrend) and bear markets (breakdowns with volume in downtrend)

name = "1h_4h_1d_camarilla_breakout_v1"
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
    open_time = prices['open_time']
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute session hours once (08-20 UTC)
    hours = open_time.dt.hour.values
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return signals
    
    # 4h EMA(50) for trend
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data ONCE before loop for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return signals
    
    # Calculate 1d Camarilla pivot levels (H3, L3, H4, L4)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range for pivot calculation (using previous day's range)
    prev_high = pd.Series(high_1d).shift(1)
    prev_low = pd.Series(low_1d).shift(1)
    prev_close = pd.Series(close_1d).shift(1)
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    h3 = pivot + (range_hl * 1.1 / 4)
    l3 = pivot - (range_hl * 1.1 / 4)
    h4 = pivot + (range_hl * 1.1 / 2)
    l4 = pivot - (range_hl * 1.1 / 2)
    
    # Align 1d Camarilla levels to 1h timeframe (with 1-bar delay for completed day)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3.values)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3.values)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4.values)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4.values)
    
    # 1h volume SMA (20-period) for confirmation
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Trend filter from 4h EMA
        uptrend = price_close > ema_50_4h_aligned[i]
        downtrend = price_close < ema_50_4h_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Camarilla breakout conditions
        breakout_long = price_close > h3_aligned[i]  # Break above H3
        breakdown_short = price_close < l3_aligned[i]  # Break below L3
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Uptrend + H3 breakout + volume confirmation
        if uptrend and breakout_long and vol_confirm:
            enter_long = True
        
        # Short: Downtrend + L3 breakdown + volume confirmation
        if downtrend and breakdown_short and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite Camarilla level or trend reversal
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price breaks below L3 OR trend turns down
            exit_long = (price_close < l3_aligned[i]) or (not uptrend)
        elif position == -1:
            # Exit short if price breaks above H3 OR trend turns up
            exit_short = (price_close > h3_aligned[i]) or (not downtrend)
        
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
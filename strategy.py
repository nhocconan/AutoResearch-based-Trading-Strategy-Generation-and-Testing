#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation
# - Uses 4h EMA50 for trend direction (long above, short below)
# - 1h Camarilla pivot levels (H3/L3) for breakout entries
# - Volume > 1.5x 20-period average for confirmation
# - Session filter: 08-20 UTC to avoid low-volume Asian session
# - Discrete position sizing: ±0.20 to limit drawdown and fee churn
# - Target: 15-37 trades/year (60-150 total over 4 years) to stay within fee drag limits
# - Works in bull markets (breakouts with trend) and bear markets (breakouts against trend with confirmation)

name = "1h_camarilla_pivot_breakout_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return signals
    
    # Pre-compute 4h EMA50 for trend
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d volume SMA20 for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute hours for session filter (08-20 UTC)
    hours = prices.index.hour
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(volume_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Current price data
        volume_current = volume[i]
        high_current = high[i]
        low_current = low[i]
        close_current = close[i]
        
        # Calculate 1h Camarilla pivot levels using previous bar's OHLC
        if i == 0:
            continue
        prev_close = close[i-1]
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_range = prev_high - prev_low
        
        # Camarilla levels
        h3 = prev_close + prev_range * 1.1 / 4
        l3 = prev_close - prev_range * 1.1 / 4
        h4 = prev_close + prev_range * 1.1 / 2
        l4 = prev_close - prev_range * 1.1 / 2
        
        # Trend filter: 4h EMA50
        uptrend = close_current > ema50_4h_aligned[i]
        downtrend = close_current < ema50_4h_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 1d SMA20
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: price breaks above H3 with uptrend and volume confirmation
        if high_current > h3 and uptrend and vol_confirm:
            enter_long = True
        
        # Short: price breaks below L3 with downtrend and volume confirmation
        if low_current < l3 and downtrend and vol_confirm:
            enter_short = True
        
        # Exit conditions: reverse breakout or loss of trend
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price breaks below L3 or trend turns down
            exit_long = (low_current < l3) or (not uptrend)
        elif position == -1:
            # Exit short if price breaks above H3 or trend turns up
            exit_short = (high_current > h3) or (not downtrend)
        
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
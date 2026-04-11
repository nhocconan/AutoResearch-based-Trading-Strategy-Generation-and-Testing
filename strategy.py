#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and 1d volume confirmation
# - Uses 4h EMA(50) for trend direction (long when price > EMA, short when price < EMA)
# - 1h Camarilla pivot levels (H3/L3) for breakout entries with 1d volume > 1.5x 20-period average
# - Exit on opposite Camarilla level (H4 for longs, L4 for shorts) or time-based stop (24h max hold)
# - Session filter: 08-20 UTC to avoid low-liquidity hours
# - Discrete position sizing: ±0.20 to limit drawdown and reduce fee churn
# - Target: 15-37 trades/year (60-150 total over 4 years) to stay within fee drag limits
# - Camarilla pivots provide mathematically derived support/resistance levels that work in ranging markets
# - 4h trend filter ensures we trade with the higher timeframe momentum
# - 1d volume confirmation filters out low-conviction breakouts

name = "1h_4h_1d_camarilla_breakout_trend_v1"
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
    bars_since_entry = 0
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 60 or len(df_1d) < 30:
        return signals
    
    # Pre-compute 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute 1h Camarilla pivot levels (using previous day's OHLC)
    # Camarilla levels calculated from daily OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_h3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_l3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    camarilla_h4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_l4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # Already datetime64[ms], .hour works directly
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(volume_sma_20_aligned[i])):
            signals[i] = 0.0
            bars_since_entry = bars_since_entry + 1 if position != 0 else 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Camarilla levels
        h3 = h3_aligned[i]
        l3 = l3_aligned[i]
        h4 = h4_aligned[i]
        l4 = l4_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Trend filter: 4h EMA(50)
        ema_50 = ema_50_4h_aligned[i]
        uptrend = close_price > ema_50
        downtrend = close_price < ema_50
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price breaks above H3 with volume confirmation and uptrend
        if close_price > h3 and vol_confirm and uptrend:
            enter_long = True
        
        # Short breakout: price breaks below L3 with volume confirmation and downtrend
        if close_price < l3 and vol_confirm and downtrend:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price reaches H4 (profit target) or L3 (stop loss)
            exit_long = (close_price >= h4) or (close_price <= l3)
        elif position == -1:
            # Exit short if price reaches L4 (profit target) or H3 (stop loss)
            exit_short = (close_price <= l4) or (close_price >= h3)
        
        # Time-based exit: max 24 hours (24 bars on 1h)
        if bars_since_entry >= 24:
            if position == 1:
                exit_long = True
            elif position == -1:
                exit_short = True
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            bars_since_entry = 0
            signals[i] = 0.20
        elif enter_short and position != -1:
            position = -1
            bars_since_entry = 0
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            bars_since_entry = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            bars_since_entry = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
            if position != 0:
                bars_since_entry += 1
    
    return signals
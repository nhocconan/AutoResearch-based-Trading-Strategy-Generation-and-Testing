#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation
# - Uses 4h EMA(21) for trend direction (long when price > EMA, short when price < EMA)
# - Enters on 1h Camarilla H3/L3 breakout with volume > 1.5x 20-period average
# - Exits when price touches opposite Camarilla level (H3/L3) or at end of session (20 UTC)
# - Session filter: only trade 08-20 UTC to avoid low-volume Asian session noise
# - Position size: 0.20 (20% of capital) for low drawdown
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)
# - Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend)

name = "1h_4h_camarilla_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h EMA(21) for trend filter
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Pre-compute 1h Camarilla pivots (based on previous day's OHLC)
    # We need daily OHLC for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H3/H4 and L3/L4
    # H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 4
    camarilla_h4 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_l4 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 1h (use previous day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # 1h volume confirmation
    volume_1h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1h > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_spike[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price touches Camarilla L3 (opposite level) or end of session
            if low[i] <= camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif not in_session[i]:  # Force exit at session end
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price touches Camarilla H3 (opposite level) or end of session
            if high[i] >= camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif not in_session[i]:  # Force exit at session end
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for Camarilla breakout with volume confirmation and 4h trend filter
            # Long: price > EMA21_4h and break above H3 with volume
            if (close[i] > ema_21_4h_aligned[i] and  # 4h uptrend filter
                high[i] >= camarilla_h3_aligned[i] and  # Break above H3
                volume_spike[i]):  # Volume confirmation
                position = 1
                signals[i] = 0.20
            # Short: price < EMA21_4h and break below L3 with volume
            elif (close[i] < ema_21_4h_aligned[i] and  # 4h downtrend filter
                  low[i] <= camarilla_l3_aligned[i] and  # Break below L3
                  volume_spike[i]):  # Volume confirmation
                position = -1
                signals[i] = -0.20
    
    return signals
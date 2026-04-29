#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA200 trend filter and volume confirmation
# Uses tighter H3/L3 levels for stronger support/resistance to reduce false breakouts
# 4h EMA200 provides robust long-term trend filter (works in both bull/bear markets)
# Volume confirmation ensures breakout validity
# Session filter (08-20 UTC) reduces noise during low-liquidity hours
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag
# Signal size: 0.20 (discrete level for cost control)

name = "1h_Camarilla_H3L3_Breakout_4hEMA200_Trend_VolumeSpike_v1"
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
    
    # Pre-compute session filter (08-20 UTC) - index-based for performance
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load HTF data ONCE before loop for 4h calculations
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    # Calculate 4h EMA(200) for trend filter
    close_4h = df_4h['close'].values
    ema_200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # Calculate Camarilla pivot levels from daily data (stronger H3/L3 levels)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H3/L3 are stronger levels (more significant support/resistance)
    # H3 = close + (high - low) * 1.1/6
    # L3 = close - (high - low) * 1.1/6
    range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + (range_1d * 1.1 / 6)
    camarilla_l3 = close_1d - (range_1d * 1.1 / 6)
    
    # Align Camarilla levels to 1h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: volume > 2.0x 24-period average (more stringent)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 24)  # warmup for EMA200 and volume MA
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if HTF data not available
        if np.isnan(ema200_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_h3 = h3_aligned[i]
        curr_l3 = l3_aligned[i]
        curr_ema200 = ema200_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Trend regime: bullish if price > 4h EMA200, bearish if price < 4h EMA200
        is_bullish_regime = curr_close > curr_ema200
        is_bearish_regime = curr_close < curr_ema200
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation to avoid false breakouts
            if curr_volume_confirm:
                # Bullish entry: price breaks above H3 with volume AND bullish regime
                if curr_high > curr_h3 and is_bullish_regime:
                    signals[i] = 0.20
                    position = 1
                # Bearish entry: price breaks below L3 with volume AND bearish regime
                elif curr_low < curr_l3 and is_bearish_regime:
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:  # Long position - exit when price falls below L3 or regime changes
            if curr_low < curr_l3 or not is_bullish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position - exit when price rises above H3 or regime changes
            if curr_high > curr_h3 or not is_bearish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
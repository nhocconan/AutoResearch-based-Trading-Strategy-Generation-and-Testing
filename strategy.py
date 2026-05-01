#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla H3/L3 breakout with 1w EMA50 trend filter and volume confirmation
# Trades breakouts at Camarilla H3 (resistance) for longs and L3 (support) for shorts
# Only trades in direction of 1w EMA50 trend to avoid counter-trend whipsaws
# Volume confirmation > 1.8x 20-period EMA ensures institutional participation
# Designed for very low trade frequency: ~8-15 trades/year per symbol with 0.30 sizing
# Works in bull/bear markets by following the 1w trend direction via EMA50

name = "1d_Camarilla_H3L3_Breakout_1wEMA50_Trend_Volume_v1"
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
    
    # 1w HTF data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for calculations)
    start_idx = 60
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels for today (using yesterday's OHLC)
        if i == 0:
            signals[i] = 0.0
            continue
            
        # Yesterday's OHLC for Camarilla calculation
        y_high = high[i-1]
        y_low = low[i-1]
        y_close = close[i-1]
        
        # Camarilla levels
        rng = y_high - y_low
        if rng <= 0:
            signals[i] = 0.0
            continue
            
        # H3 and L3 levels (most important for breakouts)
        h3 = y_close + (rng * 1.1 / 4)
        l3 = y_close - (rng * 1.1 / 4)
        
        # Volume confirmation: volume > 1.8 * 20-period EMA
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_spike = volume[i] > (1.8 * vol_ma_20)
        else:
            volume_spike = False
        
        # Trend filter: only trade in direction of 1w EMA50
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if uptrend:
                # Long: price breaks above H3 with volume spike
                if close[i] > h3 and volume_spike:
                    signals[i] = 0.30
                    position = 1
                else:
                    signals[i] = 0.0
            elif downtrend:
                # Short: price breaks below L3 with volume spike
                if close[i] < l3 and volume_spike:
                    signals[i] = -0.30
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid sideways markets
        
        elif position == 1:  # Long position
            # Exit: price closes below yesterday's close (mean reversion)
            if close[i] < y_close:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: price closes above yesterday's close (mean reversion)
            if close[i] > y_close:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and session filter
# - Uses Camarilla pivot levels (H3/L3) from prior 4h bar for breakout entries
# - 4h close vs EMA20 determines trend direction (bullish/bearish)
# - Only trade during 08-20 UTC session to avoid low-liquidity hours
# - Volume confirmation: 1h volume > 1.3x 20-period volume SMA
# - Position sizing: 0.20 discrete level to minimize fee drag
# - Target: 15-37 trades/year on 1h timeframe to stay within fee drag limits
# - Works in bull/bear: trend filter adapts direction, session avoids chop

name = "1h_4h_camarilla_breakout_trend_v1"
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
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 1h volume SMA for regime filter
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h Camarilla pivot levels (based on prior 4h bar)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    pivot = (high_4h + low_4h + close_4h) / 3.0
    range_4h = high_4h - low_4h
    camarilla_h3 = pivot + range_4h * 1.1 / 4.0
    camarilla_l3 = pivot - range_4h * 1.1 / 4.0
    
    # AlCamarilla levels to 1h timeframe (completed 4h bar only)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
    # Calculate 4h EMA20 for trend filter
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate 4h close for trend comparison
    close_4h_aligned = align_htf_to_ltf(prices, df_4h, close_4h)
    
    for i in range(40, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(close_4h_aligned[i]) or
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1h volume > 1.3x 20-period volume SMA
        vol_confirm = volume[i] > 1.3 * volume_sma_20[i]
        
        # Trend filter: 4h close vs 4h EMA20
        trend_bullish = close_4h_aligned[i] > ema_20_4h_aligned[i]
        trend_bearish = close_4h_aligned[i] < ema_20_4h_aligned[i]
        
        # Camarilla breakout signals (using prior completed 4h bar levels)
        breakout_up = close[i] > camarilla_h3_aligned[i-1]  # Break above H3
        breakout_down = close[i] < camarilla_l3_aligned[i-1]  # Break below L3
        
        # Exit conditions: opposite breakout or loss of volume confirmation
        exit_long = breakout_down or not vol_confirm
        exit_short = breakout_up or not vol_confirm
        
        if position == 0:  # Flat - look for entry
            if breakout_up and trend_bullish and vol_confirm:
                position = 1
                signals[i] = 0.20
            elif breakout_down and trend_bearish and vol_confirm:
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
    
    return signals
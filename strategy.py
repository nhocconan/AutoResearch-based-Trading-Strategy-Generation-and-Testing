#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend and volume confirmation
# - Uses 4h EMA(50) for trend filter (uptrend: close > EMA50, downtrend: close < EMA50)
# - 1h Camarilla levels (H3, L3) from previous 4h bar for breakout entries
# - Volume confirmation: 1h volume > 2.0x 20-bar average
# - Session filter: 08-20 UTC to avoid low-liquidity hours
# - Discrete position sizing (0.20) to minimize fee churn
# - Target: 20-40 trades/year (80-160 over 4 years) to avoid fee drag
# - 4h trend filter reduces false breakouts in ranging markets

name = "1h_4h_camarilla_breakout_volume_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Pre-compute 4h indicators
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Pre-compute 1h indicators
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    volume_1h = prices['volume'].values
    
    # 1h volume confirmation: > 2.0x 20-period average
    avg_volume_20_1h = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    vol_spike_1h = volume_1h > (2.0 * avg_volume_20_1h)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or 
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Calculate Camarilla levels from previous completed 4h bar
            # Need at least one completed 4h bar (16 1h bars = 1 4h bar)
            if i >= 16:
                # Get index of last completed 4h bar
                # Each 4h bar = 16 1h bars, so last completed 4h bar ends at index: ((i // 16) * 16) - 1
                htf_bar_end = ((i // 16) * 16) - 1
                if htf_bar_end >= 16:  # Need at least 16 bars for calculation
                    lookback_start = htf_bar_end - 15  # 16 bars for the completed 4h bar
                    lookback_end = htf_bar_end + 1     # exclusive
                    
                    if lookback_start >= 0 and lookback_end <= len(prices):
                        # Calculate Camarilla levels for the completed 4h bar
                        high_4h_bar = high_1h[lookback_start:lookback_end].max()
                        low_4h_bar = low_1h[lookback_start:lookback_end].min()
                        close_4h_bar = close_1h[lookback_start:lookback_end].mean()
                        
                        range_4h = high_4h_bar - low_4h_bar
                        if range_4h > 0:
                            # Camarilla levels
                            h3 = close_4h_bar + (range_4h * 1.1 / 4)
                            l3 = close_4h_bar - (range_4h * 1.1 / 4)
                            
                            # Long signal: price breaks above H3 in 4h uptrend with volume spike
                            if (close_1h[i] > h3 and 
                                close_1h[i] > ema_50_4h_aligned[i] and 
                                vol_spike_1h[i]):
                                position = 1
                                signals[i] = 0.20
                            # Short signal: price breaks below L3 in 4h downtrend with volume spike
                            elif (close_1h[i] < l3 and 
                                  close_1h[i] < ema_50_4h_aligned[i] and 
                                  vol_spike_1h[i]):
                                position = -1
                                signals[i] = -0.20
        
        elif position == 1:  # Long position - exit on reversal or volume dry-up
            # Exit if price crosses below EMA50 (trend change) or volume drops
            if close_1h[i] < ema_50_4h_aligned[i] or not vol_spike_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position - exit on reversal or volume dry-up
            # Exit if price crosses above EMA50 (trend change) or volume drops
            if close_1h[i] > ema_50_4h_aligned[i] or not vol_spike_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
    
    return signals
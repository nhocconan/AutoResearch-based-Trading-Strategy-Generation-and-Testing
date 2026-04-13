#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Camarilla pivot breakout with 1w trend filter and volume confirmation
    # Long when: price breaks above H3 (strong resistance) AND 1w EMA20 uptrend AND volume > 1.5x 20-period MA
    # Short when: price breaks below L3 (strong support) AND 1w EMA20 downtrend AND volume > 1.5x 20-period MA
    # Exit when: price returns to pivot point (mean reversion) OR adverse 1w EMA20 crossover
    # Uses discrete sizing (0.25) targeting 30-100 trades over 4 years.
    # Works in bull/bear via 1w EMA20 trend filter preventing counter-trend trades.
    # Camarilla levels provide precise intraday support/resistance based on prior day's range.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculations (yesterday's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Get 1w data for EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate volume MA20 for confirmation
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(volume_ma20[i]) or i >= len(df_1d)):
            signals[i] = 0.0
            continue
        
        # Get yesterday's OHLC for Camarilla calculation (1d bar at index i-1 in 1d timeframe)
        if i-1 < len(df_1d):
            prev_high = df_1d['high'].iloc[i-1]
            prev_low = df_1d['low'].iloc[i-1]
            prev_close = df_1d['close'].iloc[i-1]
            
            # Calculate Camarilla levels for today based on yesterday's range
            range_val = prev_high - prev_low
            if range_val <= 0:
                signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
                continue
                
            # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
            # H3 = close + 1.1 * range / 2 (strong resistance)
            # L3 = close - 1.1 * range / 2 (strong support)
            h3 = prev_close + 1.1 * range_val / 2
            l3 = prev_close - 1.1 * range_val / 2
            pivot = (prev_high + prev_low + prev_close) / 3
            
            # Volume confirmation: current volume > 1.5x 20-period MA
            volume_confirm = volume[i] > 1.5 * volume_ma20[i]
            
            # 1w EMA20 trend filter
            uptrend = ema20_1w_aligned[i] > ema20_1w_aligned[i-1] if i > 0 else False
            downtrend = ema20_1w_aligned[i] < ema20_1w_aligned[i-1] if i > 0 else False
            
            # Entry conditions
            long_entry = (close[i] > h3) and volume_confirm and uptrend and position != 1
            short_entry = (close[i] < l3) and volume_confirm and downtrend and position != -1
            
            # Exit conditions: return to pivot (mean reversion) or trend change
            exit_long = (position == 1) and (close[i] <= pivot or not uptrend)
            exit_short = (position == -1) and (close[i] >= pivot or not downtrend)
            
            # Execute signals
            if long_entry:
                position = 1
                signals[i] = position_size
            elif short_entry:
                position = -1
                signals[i] = -position_size
            elif position == 1 and exit_long:
                position = 0
                signals[i] = 0.0
            elif position == -1 and exit_short:
                position = 0
                signals[i] = 0.0
            # Hold current position
            else:
                if position == 1:
                    signals[i] = position_size
                elif position == -1:
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
        else:
            # Hold position if we don't have yesterday's data yet
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_camarilla_breakout_trend_volume_v1"
timeframe = "1d"
leverage = 1.0
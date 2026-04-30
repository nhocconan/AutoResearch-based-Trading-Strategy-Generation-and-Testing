#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R4/S4 breakout with 1w EMA50 trend filter and volume spike confirmation
# Camarilla R4/S4 levels (extreme support/resistance) break only on strong momentum with institutional volume
# 1w EMA > 50-period ensures alignment with weekly trend to avoid false breakouts in ranging markets
# Volume spike (2.5x 20-period average) confirms participation. Discrete sizing 0.28 balances return/drawdown.
# Works in bull markets via breakouts above R4 and bear markets via breakdowns below S4 with trend filter.
# Target: 50-100 total trades over 4 years (12-25/year) to minimize fee drag.

name = "1d_Camarilla_R4S4_1wEMA50_VolumeSpike_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1w data ONCE before loop (MTF Rule #1)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Camarilla pivot levels (R4, S4 - extreme levels)
    high_1d = df_1w['high'].values  # Use weekly high/low for more significant levels
    low_1d = df_1w['low'].values
    close_1w_for_camarilla = df_1w['close'].values
    
    # Camarilla R4/S4: close ± (high-low) * 1.50
    camarilla_r4 = close_1w_for_camarilla + ((high_1d - low_1d) * 1.50)
    camarilla_s4 = close_1w_for_camarilla - ((high_1d - low_1d) * 1.50)
    
    # Align Camarilla levels to 1d timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    # Volume confirmation: volume > 2.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(60, 20)  # warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema = ema_50_1w_aligned[i]
        curr_r4 = camarilla_r4_aligned[i]
        curr_s4 = camarilla_s4_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and price above/below weekly EMA for trend alignment
            if curr_volume_spike:
                # Bullish entry: break above R4 with price above weekly EMA
                if curr_close > curr_r4 and curr_close > curr_ema:
                    signals[i] = 0.28
                    position = 1
                    entry_price = curr_close
                # Bearish entry: break below S4 with price below weekly EMA
                elif curr_close < curr_s4 and curr_close < curr_ema:
                    signals[i] = -0.28
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below R4 (breakout fails) OR price crosses below weekly EMA
            if curr_close < curr_r4 or curr_close < curr_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        
        elif position == -1:  # Short position
            # Exit when price rises above S4 (breakdown fails) OR price crosses above weekly EMA
            if curr_close > curr_s4 or curr_close > curr_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals
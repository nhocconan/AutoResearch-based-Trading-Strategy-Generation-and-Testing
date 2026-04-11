#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot reversal with 1w trend filter and volume confirmation
# Long when price touches or breaks below Camarilla L3 (support) in uptrend with volume spike
# Short when price touches or breaks above Camarilla H3 (resistance) in downtrend with volume spike
# Exit when price reaches opposite H3/L3 level or trend reverses
# Designed for 10-30 trades/year on 1d timeframe with mean reversion in trending markets
# Works in bull/bear by using 1w trend to filter direction and Camarilla for precise reversal points

name = "1d_1w_camarilla_reversal_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1w EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate 20-period average volume for volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate daily Camarilla levels (based on previous day)
    # Camarilla: H4 = C + 1.5*(H-L), H3 = C + 1.125*(H-L), L3 = C - 1.125*(H-L), L4 = C - 1.5*(H-L)
    # where C = (H+L+C)/3 of previous period
    shift_high = np.roll(high, 1)
    shift_low = np.roll(low, 1)
    shift_close = np.roll(close, 1)
    # First value will be invalid (rolled from end), handled by NaN checks
    typical_price = (shift_high + shift_low + shift_close) / 3
    range_val = shift_high - shift_low
    camarilla_h3 = typical_price + 1.125 * range_val
    camarilla_l3 = typical_price - 1.125 * range_val
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.8x 20-period average
        volume_filter = volume[i] > 1.8 * vol_ma_20[i]
        
        # Trend filter: price relative to 1w EMA20
        is_uptrend = close[i] > ema_20_1w_aligned[i]
        is_downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Entry conditions: price touches/breaks Camarilla levels with volume in trend direction
        long_entry = (close[i] <= camarilla_l3[i] or low[i] <= camarilla_l3[i]) and volume_filter and is_uptrend
        short_entry = (close[i] >= camarilla_h3[i] or high[i] >= camarilla_h3[i]) and volume_filter and is_downtrend
        
        # Exit conditions: price reaches opposite Camarilla level or trend reverses
        long_exit = (close[i] >= camarilla_h3[i] or high[i] >= camarilla_h3[i]) or (not is_uptrend)
        short_exit = (close[i] <= camarilla_l3[i] or low[i] <= camarilla_l3[i]) or (not is_downtrend)
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with volume confirmation and 1w trend filter
# - Long: price breaks above Camarilla H3 level, volume > 2.0x 20-period average, price > 1w EMA(50)
# - Short: price breaks below Camarilla L3 level, volume > 2.0x 20-period average, price < 1w EMA(50)
# - Exit: price returns to Camarilla pivot point (PP)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 15-25 trades/year (60-100 total over 4 years) to stay within fee drag limits
# - Camarilla levels provide institutional support/resistance; volume confirms breakout strength; weekly trend ensures directional bias

name = "1d_1w_camarilla_breakout_volume_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1w data ONCE before loop for trend filter (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute Camarilla levels from previous day's OHLC
    # PP = (H + L + C) / 3
    # H3 = PP + (H - L) * 1.1/2
    # L3 = PP - (H - L) * 1.1/2
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    pp = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    h3 = pp + (range_hl * 1.1 / 2)
    l3 = pp - (range_hl * 1.1 / 2)
    
    # Pre-compute volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(h3[i]) or np.isnan(l3[i]) or np.isnan(pp[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume_current > 2.0 * volume_sma_20[i]
        
        # Weekly trend filter
        weekly_bias_long = close_price > ema_50_1w_aligned[i]
        weekly_bias_short = close_price < ema_50_1w_aligned[i]
        
        # Camarilla levels
        h3_level = h3[i]
        l3_level = l3[i]
        pp_level = pp[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price above H3, volume confirmation, weekly long bias
        if close_price > h3_level and vol_confirm and weekly_bias_long:
            enter_long = True
        
        # Short breakout: price below L3, volume confirmation, weekly short bias
        if close_price < l3_level and vol_confirm and weekly_bias_short:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to pivot point
            exit_long = close_price <= pp_level
        elif position == -1:
            # Exit short if price returns to pivot point
            exit_short = close_price >= pp_level
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
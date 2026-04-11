#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with volume confirmation and weekly trend filter
# - Long: price breaks above H3 level, volume > 1.5x 20-day avg, price > weekly EMA(20)
# - Short: price breaks below L3 level, volume > 1.5x 20-day avg, price < weekly EMA(20)
# - Exit: price returns to pivot point (PP) or opposite Camarilla level (L3/H3)
# - Uses 1w EMA(20) for trend filter to avoid counter-trend trades
# - Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag
# - Camarilla levels provide institutional support/resistance that work in ranging and trending markets

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
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return signals
    
    # Pre-compute weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Pre-compute daily Camarilla pivot levels from previous day
    # PP = (H + L + C) / 3
    # R4 = PP + (H - L) * 1.1/2, R3 = PP + (H - L) * 1.1/4, R2 = PP + (H - L) * 1.1/6, R1 = PP + (H - L) * 1.1/12
    # S1 = PP - (H - L) * 1.1/12, S2 = PP - (H - L) * 1.1/6, S3 = PP - (H - L) * 1.1/4, S4 = PP - (H - L) * 1.1/2
    # Camarilla uses H3 = R3 and L3 = S3 for trading
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # First bar: use current values
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    pp = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    h3 = pp + range_hl * 1.1 / 4  # R3
    l3 = pp - range_hl * 1.1 / 4  # S3
    
    # Pre-compute daily volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(h3[i]) or np.isnan(l3[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Weekly trend filter
        weekly_uptrend = close_price > ema_20_1w_aligned[i]
        weekly_downtrend = close_price < ema_20_1w_aligned[i]
        
        # Camarilla levels
        h3_level = h3[i]
        l3_level = l3[i]
        pp_level = pp[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price closes above H3, volume confirmation, weekly uptrend
        if close_price > h3_level and vol_confirm and weekly_uptrend:
            enter_long = True
        
        # Short breakout: price closes below L3, volume confirmation, weekly downtrend
        if close_price < l3_level and vol_confirm and weekly_downtrend:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to pivot point or below L3
            exit_long = close_price <= pp_level
        elif position == -1:
            # Exit short if price returns to pivot point or above H3
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
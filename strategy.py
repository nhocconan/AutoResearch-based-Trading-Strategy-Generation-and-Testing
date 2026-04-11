# 4h_1d_camarilla_breakout_volume_v1
# Hypothesis: Camarilla pivot levels from 1d combined with volume confirmation and 4h trend filter
# - Long when price breaks above S3 with volume > 1.5x average and price > 4h EMA50
# - Short when price breaks below S4 with volume > 1.5x average and price < 4h EMA50
# - Uses Camarilla levels calculated from previous day's high/low/close
# - Volume confirmation reduces false breakouts
# - EMA50 trend filter ensures trades align with intermediate trend
# - Designed for 4-8 trades per month per symbol (48-96/year) to stay within fee limits
# - Works in bull markets (breakouts continue) and bear markets (breakdowns continue)

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return signals
    
    # Calculate Camarilla levels from previous day's data
    # Camarilla formulas:
    # H4 = Close + 1.5 * (High - Low)
    # H3 = Close + 1.0 * (High - Low)
    # H2 = Close + 0.5 * (High - Low)
    # H1 = Close + 0.25 * (High - Low)
    # L1 = Close - 0.25 * (High - Low)
    # L2 = Close - 0.5 * (High - Low)
    # L3 = Close - 1.0 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    
    prev_high = df_1d['high'].shift(1).values  # Previous day's high
    prev_low = df_1d['low'].shift(1).values    # Previous day's low
    prev_close = df_1d['close'].shift(1).values # Previous day's close
    
    # Calculate Camarilla levels
    camarilla_h4 = prev_close + 1.5 * (prev_high - prev_low)
    camarilla_h3 = prev_close + 1.0 * (prev_high - prev_low)
    camarilla_l3 = prev_close - 1.0 * (prev_high - prev_low)
    camarilla_l4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Pre-compute 4h EMA50 for trend filter
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Pre-compute 4h volume average (20-period)
    volume_series = pd.Series(volume)
    volume_avg_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(ema50[i]) or np.isnan(volume_avg_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_avg_20[i]
        
        # Trend filter: price relative to 4h EMA50
        price_above_ema50 = price_close > ema50[i]
        price_below_ema50 = price_close < ema50[i]
        
        # Camarilla breakout conditions
        # Long when price breaks above H3 with volume and trend alignment
        breakout_long = price_high > camarilla_h3_aligned[i]
        # Short when price breaks below L3 with volume and trend alignment
        breakdown_short = price_low < camarilla_l3_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Breakout above H3 + volume confirmation + price above EMA50
        if breakout_long and vol_confirm and price_above_ema50:
            enter_long = True
        
        # Short: Breakdown below L3 + volume confirmation + price below EMA50
        if breakdown_short and vol_confirm and price_below_ema50:
            enter_short = True
        
        # Exit conditions: opposite breakout or trend reversal
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if breakdown below L3 OR price falls below EMA50
            exit_long = breakdown_short or (not price_above_ema50)
        elif position == -1:
            # Exit short if breakout above H3 OR price rises above EMA50
            exit_short = breakout_long or (not price_below_ema50)
        
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
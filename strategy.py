# 1d_1w_camarilla_volume_trend_v1
# Hypothesis: Daily Camarilla pivot levels + weekly trend filter + volume confirmation
# - Uses Camarilla pivot levels (H3/L3) from daily timeframe for mean-reversion entries
# - Weekly trend filter (price vs weekly EMA20) to align with higher timeframe trend
# - Volume confirmation (current volume > 1.5x 20-period average) to avoid low-volume false signals
# - Long when price crosses below L3 (oversold) in uptrend, short when price crosses above H3 (overbought) in downtrend
# - Exits when price reverts to daily pivot (central level) or opposite Camarilla level is touched
# - Designed for 1d timeframe with ~15-30 trades/year to stay within fee drag limits
# - Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_volume_trend_v1"
timeframe = "1d"
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
    
    # Load daily data for Camarilla pivots (use same timeframe data)
    # Since we're on 1d timeframe, we can calculate pivots directly
    
    # Load weekly data for trend filter ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return signals
    
    # Pre-compute weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Pre-compute daily volume SMA (20-period)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(20, n):  # Start after 20-bar warmup for weekly EMA
        # Skip if any required data is invalid
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Calculate daily Camarilla pivot levels for today
        # Need yesterday's OHLC to calculate today's levels
        if i == 0:
            # Not enough data for yesterday
            signals[i] = 0.0
            continue
            
        # Use previous day's OHLC to calculate today's Camarilla levels
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Camarilla pivot calculations
        # Pivot = (High + Low + Close) / 3
        pivot = (prev_high + prev_low + prev_close) / 3
        # Range = High - Low
        range_val = prev_high - prev_low
        
        # Camarilla levels
        # H4 = Close + Range * 1.1/2
        # H3 = Close + Range * 1.1/4
        # L3 = Close - Range * 1.1/4
        # L4 = Close - Range * 1.1/2
        h3 = prev_close + range_val * 1.1 / 4
        l3 = prev_close - range_val * 1.1 / 4
        
        # Current price data
        price_close = close[i]
        volume_current = volume[i]
        
        # Weekly trend filter
        price_above_weekly_ema = price_close > ema20_1w_aligned[i]
        price_below_weekly_ema = price_close < ema20_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Price crosses below L3 (oversold) in uptrend + volume confirmation
        if price_close < l3 and price_above_weekly_ema and vol_confirm:
            enter_long = True
        
        # Short: Price crosses above H3 (overbought) in downtrend + volume confirmation
        if price_close > h3 and price_below_weekly_ema and vol_confirm:
            enter_short = True
        
        # Exit conditions: price returns to pivot level or opposite extreme touched
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to pivot or touches H3 (overbought)
            exit_long = price_close >= pivot or price_close >= h3
        elif position == -1:
            # Exit short if price returns to pivot or touches L3 (oversold)
            exit_short = price_close <= pivot or price_close <= l3
        
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
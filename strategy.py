#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with weekly trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 level AND weekly trend is up (price > weekly EMA20)
# - Short when price breaks below Camarilla L3 level AND weekly trend is down (price < weekly EMA20)
# - Volume confirmation: 1d volume > 1.5x 20-period 1d volume SMA
# - Exit: Price returns to Camarilla pivot level (midpoint) or opposite breakout with volume
# - Position sizing: 0.25 discrete level
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# - Weekly EMA20 provides structural bias, Camarilla levels for precise entries, volume for confirmation
# - Designed to work in both bull (trend following) and bear (mean reversion at extremes) markets

name = "1d_1w_camarilla_pivot_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 20-period volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate weekly EMA20 for trend filter
    weekly_close = df_1w['close'].values
    weekly_ema20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_ema20_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema20)
    
    # Track entry price for exit logic
    entry_price = np.full(n, np.nan)
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(volume_sma_20[i]) or np.isnan(weekly_ema20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Need at least 1 day of prior data for Camarilla calculation
        if i < 1:
            signals[i] = 0.0
            continue
            
        # Calculate Camarilla pivot levels from previous day's OHLC
        # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
        # H3 = Close + 1.1*(High-Low)/2
        # L3 = Close - 1.1*(High-Low)/2
        # Pivot = (High + Low + Close)/3
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        pivot = (prev_high + prev_low + prev_close) / 3.0
        camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 2.0
        camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 2.0
        
        # Volume confirmation: 1d volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # Weekly trend filter
        weekly_trend_up = close[i] > weekly_ema20_aligned[i]
        weekly_trend_down = close[i] < weekly_ema20_aligned[i]
        
        if position == 0:  # Flat - look for entry
            # Long: break above H3 with weekly uptrend and volume confirmation
            if close[i] > camarilla_h3 and weekly_trend_up and vol_confirm:
                position = 1
                signals[i] = 0.25
                entry_price[i] = close[i]
            # Short: break below L3 with weekly downtrend and volume confirmation
            elif close[i] < camarilla_l3 and weekly_trend_down and vol_confirm:
                position = -1
                signals[i] = -0.25
                entry_price[i] = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            # Exit when price returns to pivot level or breaks below L3 with volume
            exit_condition = (close[i] <= pivot) or \
                           (close[i] < camarilla_l3 and vol_confirm)
            if exit_condition:
                position = 0
                signals[i] = 0.0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            # Exit when price returns to pivot level or breaks above H3 with volume
            exit_condition = (close[i] >= pivot) or \
                           (close[i] > camarilla_h3 and vol_confirm)
            if exit_condition:
                position = 0
                signals[i] = 0.0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
    
    return signals
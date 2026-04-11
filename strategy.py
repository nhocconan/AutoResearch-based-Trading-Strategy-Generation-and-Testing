#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout + 1d volume spike + chop regime filter
# - Camarilla levels from 1d: L3/H3 act as intraday support/resistance, L4/H4 as stronger breakout levels
# - Long when price breaks above H4 with volume > 1.8x 20-period 1d average (strong conviction)
# - Short when price breaks below L4 with volume > 1.8x 20-period 1d average
# - Chop regime filter: only trade when Choppiness Index(14) < 38.2 (trending market) to avoid false breakouts in chop
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits for 12h
# - Volume requirement (>1.8x average) ensures we only trade high-conviction breakouts
# - Works in both bull (breakouts with volume) and bear (breakdowns with volume) markets
# - 1d HTF provides reliable Camarilla calculation and volume confirmation

name = "12h_1d_camarilla_volume_chop_v3"
timeframe = "12h"
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
    
    # Load 1d data ONCE before loop for Camarilla levels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Pre-compute 1d Camarilla levels (based on previous day's range)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_h4 = np.full_like(close_1d, np.nan)
    camarilla_l4 = np.full_like(close_1d, np.nan)
    camarilla_h3 = np.full_like(close_1d, np.nan)
    camarilla_l3 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        # Use previous day's OHLC to calculate today's Camarilla levels
        high_prev = high_1d[i-1]
        low_prev = low_1d[i-1]
        close_prev = close_1d[i-1]
        
        range_prev = high_prev - low_prev
        if range_prev <= 0:
            continue
            
        camarilla_h4[i] = close_prev + 1.1 * range_prev / 2
        camarilla_l4[i] = close_prev - 1.1 * range_prev / 2
        camarilla_h3[i] = close_prev + 1.1 * range_prev / 4
        camarilla_l3[i] = close_prev - 1.1 * range_prev / 4
    
    # Pre-compute 1d volume SMA (20-period)
    volume_1d = df_1d['volume'].values
    volume_series = pd.Series(volume_1d)
    volume_sma_20_1d = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute 1d Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(TR14) / (ATR14 * 14)) / log10(14)
    tr_list = []
    for i in range(len(close_1d)):
        if i == 0:
            tr = high_1d[i] - low_1d[i]
        else:
            tr = max(high_1d[i] - low_1d[i], 
                     abs(high_1d[i] - close_1d[i-1]),
                     abs(low_1d[i] - close_1d[i-1]))
        tr_list.append(tr)
    
    tr_array = np.array(tr_list)
    atr_14 = pd.Series(tr_array).rolling(window=14, min_periods=14).mean().values
    sum_tr_14 = pd.Series(tr_array).rolling(window=14, min_periods=14).sum().values
    
    chop = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if not np.isnan(atr_14[i]) and atr_14[i] > 0 and not np.isnan(sum_tr_14[i]):
            chop[i] = 100 * np.log10(sum_tr_14[i] / (atr_14[i] * 14)) / np.log10(14)
        else:
            chop[i] = np.nan
    
    # Align 1d indicators to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Camarilla breakout conditions
        breakout_long = price_close > camarilla_h4_aligned[i-1]  # Close above previous period's H4
        breakout_short = price_close < camarilla_l4_aligned[i-1]  # Close below previous period's L4
        
        # Volume confirmation: current volume > 1.8x 20-period average (using 1d aligned volume)
        vol_confirm = volume_current > 1.8 * volume_sma_20_aligned[i]
        
        # Chop regime filter: only trade when CHOP < 38.2 (trending market)
        chop_filter = chop_aligned[i] < 38.2
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Camarilla H4 breakout + volume confirmation + chop filter
        if breakout_long and vol_confirm and chop_filter:
            enter_long = True
        
        # Short: Camarilla L4 breakdown + volume confirmation + chop filter
        if breakout_short and vol_confirm and chop_filter:
            enter_short = True
        
        # Exit conditions: opposite Camarilla breakout or chop regime change
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price breaks below L3 OR chop regime becomes too high
            exit_long = (price_close < camarilla_l3_aligned[i-1]) or (chop_aligned[i] >= 38.2)
        elif position == -1:
            # Exit short if price breaks above H3 OR chop regime becomes too high
            exit_short = (price_close > camarilla_h3_aligned[i-1]) or (chop_aligned[i] >= 38.2)
        
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
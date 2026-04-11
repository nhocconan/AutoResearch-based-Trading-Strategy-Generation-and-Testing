#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with volume confirmation and chop regime filter
# - Long: price breaks above Camarilla H3 level from prior 1d, volume > 1.8x 20-period avg, CHOP(14) < 42 (trending)
# - Short: price breaks below Camarilla L3 level from prior 1d, volume > 1.8x 20-period avg, CHOP(14) < 42 (trending)
# - Exit: price returns to Camarilla pivot point (PP) or opposite H3/L3 level
# - Uses 1w EMA(20) for major trend filter: price > EMA20 for long bias, price < EMA20 for short bias
# - Target: 12-30 trades/year (50-120 total over 4 years) to stay within fee drag limits
# - Camarilla levels provide precise intraday support/resistance that works in both bull/bear markets
# - Volume confirmation ensures breakouts have conviction
# - CHOP filter avoids false signals in ranging markets
# - Weekly EMA prevents trading against major trend

name = "12h_1w_camarilla_volume_chop_v1"
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
    
    # Load 1d data ONCE before loop for Camarilla levels (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return signals
    
    # Pre-compute 1d Camarilla levels from prior day
    # Camarilla: based on prior day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each prior day
    camarilla_pp = np.zeros(len(close_1d))
    camarilla_h3 = np.zeros(len(close_1d))
    camarilla_l3 = np.zeros(len(close_1d))
    camarilla_h4 = np.zeros(len(close_1d))
    camarilla_l4 = np.zeros(len(close_1d))
    
    for i in range(len(close_1d)):
        if i == 0:
            camarilla_pp[i] = np.nan
            camarilla_h3[i] = np.nan
            camarilla_l3[i] = np.nan
            camarilla_h4[i] = np.nan
            camarilla_l4[i] = np.nan
        else:
            # Prior day's OHLC
            ph = high_1d[i-1]
            pl = low_1d[i-1]
            pc = close_1d[i-1]
            
            camarilla_pp[i] = (ph + pl + pc) / 3
            camarilla_h3[i] = camarilla_pp[i] + (ph - pl) * 1.1 / 4
            camarilla_l3[i] = camarilla_pp[i] - (ph - pl) * 1.1 / 4
            camarilla_h4[i] = camarilla_pp[i] + (ph - pl) * 1.1 / 2
            camarilla_l4[i] = camarilla_pp[i] - (ph - pl) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (wait for prior 1d close)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Load 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        # Fallback to 1d if insufficient weekly data
        df_1w = df_1d
    
    # Pre-compute 1w EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Pre-compute 12h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute Choppiness Index (CHOP) for regime detection
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (highest_high - lowest_low)))
    # Simplified: CHOP = 100 * log10(ATR_sum / (log10(period) * range))
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14 / (np.log10(14) * (highest_high_14 - lowest_low_14)))
    # Handle division by zero or invalid values
    chop = np.where((highest_high_14 - lowest_low_14) > 0, chop, 50.0)
    chop = np.where(np.isnan(chop), 50.0, chop)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(chop[i]) or np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Camarilla levels from prior 1d
        pp = camarilla_pp_aligned[i]
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        h4 = camarilla_h4_aligned[i]
        l4 = camarilla_l4_aligned[i]
        
        # Volume confirmation: current volume > 1.8x 20-period average
        vol_confirm = volume_current > 1.8 * volume_sma_20[i]
        
        # Chop regime filter: CHOP < 42 indicates trending market (lower = more trending)
        chop_trend = chop[i] < 42
        
        # 1w EMA trend bias
        ema_bias_long = close_price > ema_20_1w_aligned[i]
        ema_bias_short = close_price < ema_20_1w_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price breaks above H3 with volume and trend confirmation
        if close_price > h3 and vol_confirm and chop_trend and ema_bias_long:
            enter_long = True
        
        # Short breakout: price breaks below L3 with volume and trend confirmation
        if close_price < l3 and vol_confirm and chop_trend and ema_bias_short:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to pivot point or breaks below L3 (reversal)
            exit_long = close_price <= pp or close_price < l3
        elif position == -1:
            # Exit short if price returns to pivot point or breaks above H3 (reversal)
            exit_short = close_price >= pp or close_price > h3
        
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
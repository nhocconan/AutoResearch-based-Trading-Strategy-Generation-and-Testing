#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot levels from 1w + volume spike + choppiness regime filter
# - Long when price touches or breaks above H3 pivot level AND 1w trend is bullish (close > EMA20)
# - Short when price touches or breaks below L3 pivot level AND 1w trend is bearish (close < EMA20)
# - Volume confirmation: 1d volume > 2.0x 20-period volume SMA (spike filter)
# - Chop regime: only trade when choppiness index < 61.8 (trending market)
# - Exit: opposite pivot touch or volume drops below average
# - Position sizing: 0.25 discrete level
# - Target: 15-35 trades/year on 1d timeframe to stay within fee drag limits
# - Works in both bull/bear: pivots act as support/resistance in ranging markets,
#   while trend filter ensures we only trade with the weekly momentum

name = "1d_1w_camarilla_pivot_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 1d Camarilla pivot levels from previous day
    # PP = (H + L + C) / 3
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # R2 = C + ((H-L) * 1.1/6)
    # R1 = C + ((H-L) * 1.1/12)
    # S1 = C - ((H-L) * 1.1/12)
    # S2 = C - ((H-L) * 1.1/6)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    # We use H3 (R3) and L3 (S3) as entry levels
    
    # Shift high/low/close by 1 to use previous day's data
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    prev_close = np.concatenate([[np.nan], close[:-1]])
    
    pivot_point = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # H3 and L3 levels
    camarilla_h3 = prev_close + (range_hl * 1.1 / 4.0)
    camarilla_l3 = prev_close - (range_hl * 1.1 / 4.0)
    
    # Calculate 1w EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate 1w close for trend comparison
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Calculate 1d volume SMA for volume spike filter
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate choppiness index on 1d (using 14-period)
    # CHOP = 100 * log10(sum(ATR(1)) / (n * ATR(n))) / log10(n)
    # We'll use a simplified version: high-low range based
    hl_range = high - low
    atr_1 = pd.Series(hl_range).rolling(window=1, min_periods=1).sum()
    atr_14 = pd.Series(hl_range).rolling(window=14, min_periods=14).sum()
    chop = 100 * np.log10(atr_14 / (14 * atr_1)) / np.log10(14)
    # Handle division by zero and invalid values
    chop = np.where((atr_1 == 0) | np.isnan(atr_1) | np.isnan(atr_14), 50.0, chop)
    
    for i in range(40, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(close_1w_aligned[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 2.0x 20-period volume SMA (spike filter)
        vol_confirm = volume[i] > 2.0 * volume_sma_20[i]
        
        # Trend filter: 1w close vs 1w EMA20
        trend_bullish = close_1w_aligned[i] > ema_20_1w_aligned[i]
        trend_bearish = close_1w_aligned[i] < ema_20_1w_aligned[i]
        
        # Choppiness regime filter: only trade when CHOP < 61.8 (trending market)
        chop_filter = chop[i] < 61.8
        
        # Camarilla pivot touch/breakout signals
        # Touch or break above H3 (resistance) for long
        pivot_up = close[i] >= camarilla_h3[i]
        # Touch or break below L3 (support) for short
        pivot_down = close[i] <= camarilla_l3[i]
        
        # Exit conditions: opposite pivot touch or loss of volume confirmation
        exit_long = pivot_down or not vol_confirm
        exit_short = pivot_up or not vol_confirm
        
        if position == 0:  # Flat - look for entry
            if pivot_up and trend_bullish and vol_confirm and chop_filter:
                position = 1
                signals[i] = 0.25
            elif pivot_down and trend_bearish and vol_confirm and chop_filter:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals
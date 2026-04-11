#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout + 1d volume confirmation + chop regime filter
# - Camarilla levels from 1d: L3, L4, H3, H4 act as intraday support/resistance
# - Long when price breaks above H3 with volume > 1.2x 20-period average and chop < 61.8 (trending)
# - Short when price breaks below L3 with volume > 1.2x 20-period average and chop < 61.8 (trending)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 19-50 trades/year (75-200 total over 4 years) to stay within fee drag limits
# - Camarilla pivots work in both bull (breakouts with volume) and bear (breakdowns with volume) markets
# - 1d HTF provides reliable structure, chop filter avoids false signals in ranging markets

name = "4h_1d_camarilla_volume_chop_v2"
timeframe = "4h"
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
    
    # Load 1d data ONCE before loop for Camarilla pivots, volume, and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d Camarilla levels (based on previous day's range)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_h3 = np.zeros(len(df_1d))
    camarilla_l3 = np.zeros(len(df_1d))
    camarilla_h4 = np.zeros(len(df_1d))
    camarilla_l4 = np.zeros(len(df_1d))
    pivot = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        if i == 0:
            # First day: use same values (no look-ahead)
            camarilla_h3[i] = high_1d[i]
            camarilla_l3[i] = low_1d[i]
            camarilla_h4[i] = high_1d[i]
            camarilla_l4[i] = low_1d[i]
            pivot[i] = (high_1d[i] + low_1d[i] + close_1d[i]) / 3
        else:
            # Use previous day's OHLC to avoid look-ahead
            phigh = high_1d[i-1]
            plow = low_1d[i-1]
            pclose = close_1d[i-1]
            pivot[i] = (phigh + plow + pclose) / 3
            range_val = phigh - plow
            camarilla_h3[i] = pivot[i] + range_val * 1.1 / 4
            camarilla_l3[i] = pivot[i] - range_val * 1.1 / 4
            camarilla_h4[i] = pivot[i] + range_val * 1.1 / 2
            camarilla_l4[i] = pivot[i] - range_val * 1.1 / 2
    
    # Pre-compute 1d volume SMA (20-period)
    volume_1d = df_1d['volume'].values
    volume_series = pd.Series(volume_1d)
    volume_sma_20_1d = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute 1d chopiness index (14-period)
    # Chop = 100 * log10(sum(ATR(1) over n) / (log10(n) * (max(high) - min(low) over n)))
    tr_1d = np.maximum(np.abs(high_1d - low_1d), 
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                  np.abs(np.roll(close_1d, 1) - low_1d)))
    tr_1d[0] = high_1d[0] - low_1d[0]  # First TR
    atr_1_1d = tr_1d  # ATR(1) is just true range
    
    chop_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        sum_tr = np.sum(atr_1_1d[i-13:i+1])  # Sum of last 14 TR values
        max_high = np.max(high_1d[i-13:i+1])
        min_low = np.min(low_1d[i-13:i+1])
        if max_high > min_low:
            chop_1d[i] = 100 * np.log10(sum_tr / 14) / np.log10(max_high - min_low)
        else:
            chop_1d[i] = 50  # Neutral when range is zero
    
    # Align 1d indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Camarilla breakout conditions
        breakout_long = price_close > camarilla_h3_aligned[i-1]  # Close above previous period's H3
        breakout_short = price_close < camarilla_l3_aligned[i-1]  # Close below previous period's L3
        
        # Volume confirmation: current volume > 1.2x 20-period average
        vol_confirm = volume_current > 1.2 * volume_sma_20_aligned[i]
        
        # Chop filter: only trade when market is trending (chop < 61.8)
        chop_filter = chop_aligned[i] < 61.8
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Camarilla H3 breakout + volume confirmation + chop filter
        if breakout_long and vol_confirm and chop_filter:
            enter_long = True
        
        # Short: Camarilla L3 breakdown + volume confirmation + chop filter
        if breakout_short and vol_confirm and chop_filter:
            enter_short = True
        
        # Exit conditions: opposite Camarilla breakout or chop regime shift to ranging
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price breaks below L3 OR chop becomes ranging (chop >= 61.8)
            exit_long = (price_close < camarilla_l3_aligned[i-1]) or (chop_aligned[i] >= 61.8)
        elif position == -1:
            # Exit short if price breaks above H3 OR chop becomes ranging (chop >= 61.8)
            exit_short = (price_close > camarilla_h3_aligned[i-1]) or (chop_aligned[i] >= 61.8)
        
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
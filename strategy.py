#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout + 1d volume spike + chop regime filter
# - Camarilla levels from 1d: L3/H3 act as intraday support/resistance
# - Long when price breaks above H3 with volume > 1.8x 24-period average (strong conviction)
# - Short when price breaks below L3 with volume > 1.8x 24-period average
# - Chop regime filter: only trade when Chop(14) < 61.8 (trending market) to avoid sideways chop
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) for 12h timeframe
# - Works in both bull (breakouts with volume) and bear (breakdowns with volume) markets
# - 1d HTF provides reliable Camarilla levels and volume confirmation

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
    
    # Load 1d data ONCE before loop for Camarilla levels, volume, and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d Camarilla pivot levels (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: H3, H4, L3, L4
    camarilla_h3 = pivot + (range_1d * 1.1 / 4.0)
    camarilla_l3 = pivot - (range_1d * 1.1 / 4.0)
    camarilla_h4 = pivot + (range_1d * 1.1 / 2.0)
    camarilla_l4 = pivot - (range_1d * 1.1 / 2.0)
    
    # 1d volume SMA (24-period)
    volume_1d = df_1d['volume'].values
    volume_series = pd.Series(volume_1d)
    volume_sma_24_1d = volume_series.rolling(window=24, min_periods=24).mean().values
    
    # 1d Chopiness Index (14-period)
    # TR = max(high-low, abs(high-previous_close), abs(low-previous_close))
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(np.maximum(tr1, tr2), tr3)
    # Set first TR to high-low (no previous close)
    tr_1d[0] = high_1d[0] - low_1d[0]
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    # Chop = log10(sum(tr) / (max(high)-min(low)) over 14 periods) * 100 / log10(14)
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    denominator = max_high - min_low
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)
    chop_14_1d = np.log10(tr_sum / denominator) * 100.0 / np.log10(14)
    
    # Align 1d indicators to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    volume_sma_24_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_24_1d)
    chop_14_aligned = align_htf_to_ltf(prices, df_1d, chop_14_1d)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_sma_24_aligned[i]) or np.isnan(chop_14_aligned[i])):
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
        
        # Volume confirmation: current volume > 1.8x 24-period average (using 1d aligned volume)
        vol_confirm = volume_current > 1.8 * volume_sma_24_aligned[i]
        
        # Chop regime filter: trade only when Chop < 61.8 (trending market)
        chop_filter = chop_14_aligned[i] < 61.8
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Camarilla H3 breakout + volume confirmation + chop filter
        if breakout_long and vol_confirm and chop_filter:
            enter_long = True
        
        # Short: Camarilla L3 breakdown + volume confirmation + chop filter
        if breakout_short and vol_confirm and chop_filter:
            enter_short = True
        
        # Exit conditions: opposite Camarilla breakout or chop regime shifts to ranging
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price breaks below L3 OR chop shifts to ranging (Chop > 61.8)
            exit_long = (price_close < camarilla_l3_aligned[i-1]) or (chop_14_aligned[i] >= 61.8)
        elif position == -1:
            # Exit short if price breaks above H3 OR chop shifts to ranging (Chop > 61.8)
            exit_short = (price_close > camarilla_h3_aligned[i-1]) or (chop_14_aligned[i] >= 61.8)
        
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
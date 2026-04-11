#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with volume confirmation and choppiness regime filter
# - Long: price breaks above Camarilla H3 level, volume > 1.5x 20-period avg, CHOP(14) < 38.2 (trending)
# - Short: price breaks below Camarilla L3 level, volume > 1.5x 20-period avg, CHOP(14) < 38.2 (trending)
# - Exit: price returns to Camarilla pivot point (PP) or opposite H3/L3 level
# - Uses 1d trend filter: price > EMA(50) for long bias, price < EMA(50) for short bias
# - Target: 12-30 trades/year (50-120 total over 4 years) to stay within fee drag limits
# - Camarilla pivots provide precise intraday support/resistance levels that work in 12h timeframe
# - Volume confirmation ensures breakout validity
# - Choppiness filter avoids whipsaws in ranging markets
# - 1d EMA bias aligns with higher timeframe trend

name = "12h_1d_camarilla_breakout_chop_v2"
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
    
    # Load 1d data ONCE before loop for EMA trend filter (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA(50) for trend bias
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 12h Camarilla pivot levels (based on previous bar's OHLC)
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # We use H3 for long entry and L3 for short entry
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # First bar uses current close
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    pivot_point = (prev_high + prev_low + prev_close) / 3
    range_h_l = prev_high - prev_low
    camarilla_h3 = pivot_point + (range_h_l * 1.1 / 4)
    camarilla_l3 = pivot_point - (range_h_l * 1.1 / 4)
    
    # Pre-compute 12h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute Choppiness Index (CHOP) for regime detection
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (max(high) - min(low)))) over period
    # Simplified: CHOP(14) = 100 * log10(sum(TR(14)) / (log10(14) * (max_high_14 - min_low_14)))
    # We use a rolling implementation
    tr1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]
    atr_14 = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denom = np.log10(14) * (max_high_14 - min_low_14)
    # Avoid division by zero
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop = 100 * np.log10(atr_14 / chop_denom)
    # CHOP < 38.2 = trending, CHOP > 61.8 = ranging
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(chop[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Camarilla levels
        h3_level = camarilla_h3[i]
        l3_level = camarilla_l3[i]
        pp_level = pivot_point[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Regime filter: CHOP < 38.2 indicates trending market (avoid ranging)
        trending_regime = chop[i] < 38.2
        
        # 1d EMA trend bias
        ema_bias_long = close_price > ema_50_1d_aligned[i]
        ema_bias_short = close_price < ema_50_1d_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price above Camarilla H3, volume confirmation, trending regime, long bias
        if close_price > h3_level and vol_confirm and trending_regime and ema_bias_long:
            enter_long = True
        
        # Short breakout: price below Camarilla L3, volume confirmation, trending regime, short bias
        if close_price < l3_level and vol_confirm and trending_regime and ema_bias_short:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to pivot point or drops below L3
            exit_long = close_price <= pp_level or close_price < l3_level
        elif position == -1:
            # Exit short if price returns to pivot point or rises above H3
            exit_short = close_price >= pp_level or close_price > h3_level
        
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